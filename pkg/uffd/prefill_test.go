package uffd

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"testing"

	"github.com/kuasar-sandbox/accelerator/pkg/manifest/fetch"
	"github.com/kuasar-sandbox/accelerator/pkg/sparse"
	"golang.org/x/sys/unix"
)

// prefillReader wraps a fake stream in the production SnapshotReader so the
// prefill walk exercises the same run semantics as the fault path.
func prefillReader(t *testing.T, stream fetch.Stream, ramSize uint64) SnapshotReader {
	t.Helper()
	reader, err := NewStreamSnapshotSource(stream, ramSize)
	if err != nil {
		t.Fatal(err)
	}
	return reader
}

// prefillStream is a fetch.Stream-shaped fake over a fixed run table, giving
// prefill tests full control over extent layout (data / zero / hole mixes).
type prefillStream struct {
	size uint64
	runs []prefillFakeRun
}

type prefillFakeRun struct {
	offset, end uint64
	kind        sparse.RunKind
	fill        byte
}

func (s *prefillStream) Size() uint64 { return s.size }

func (s *prefillStream) RunAt(offset, limit uint64) (sparse.Run, error) {
	if offset >= s.size {
		return nil, io.EOF
	}
	if limit == 0 {
		return nil, fmt.Errorf("prefill test: zero limit")
	}
	for _, r := range s.runs {
		if r.end <= offset {
			continue
		}
		if r.offset > offset {
			// Semantic gap (undeclared span): expose a zero run up to the
			// next declared run, like the real resolver would.
			end := r.offset
			if end > offset+limit {
				end = offset + limit
			}
			return prefillFakeRun{offset: offset, end: end, kind: sparse.Zero}, nil
		}
		end := r.end
		if end > offset+limit {
			end = offset + limit
		}
		return prefillFakeRun{offset: offset, end: end, kind: r.kind, fill: r.fill}, nil
	}
	end := s.size
	if end > offset+limit {
		end = offset + limit
	}
	return prefillFakeRun{offset: offset, end: end, kind: sparse.Zero}, nil
}

func (s *prefillStream) ReadAt(_ context.Context, _ []byte, _ uint64) (int, error) {
	return 0, errors.New("prefill test: stream ReadAt must not be called")
}

func (s *prefillStream) Close() error { return nil }

func (r prefillFakeRun) Offset() uint64       { return r.offset }
func (r prefillFakeRun) End() uint64          { return r.end }
func (r prefillFakeRun) Kind() sparse.RunKind { return r.kind }

func (r prefillFakeRun) ReadAt(_ context.Context, buf []byte, innerOffset uint64) (int, error) {
	length := r.end - r.offset
	if innerOffset > length || uint64(len(buf)) > length-innerOffset {
		return 0, fmt.Errorf("prefill fake run read outside [0,%d)", length)
	}
	if r.kind != sparse.Data {
		clear(buf)
		return len(buf), nil
	}
	for i := range buf {
		buf[i] = r.fill
	}
	return len(buf), nil
}

// failingStream injects RunAt failures at one declared offset.
type failingStream struct {
	prefillStream
	failOffset uint64
	failErr    error
}

func (s *failingStream) RunAt(offset, limit uint64) (sparse.Run, error) {
	if offset == s.failOffset {
		return nil, s.failErr
	}
	return s.prefillStream.RunAt(offset, limit)
}

type alwaysFailSource struct{}

func (alwaysFailSource) RunAt(uint64, uint64) (sparse.Run, error) {
	return nil, errors.New("broken reader")
}

func TestPrefillInstallsDataExtentsAndSkipsZero(t *testing.T) {
	const pages = 8
	// data, zero, data, hole, data layout; non-data runs must never reach
	// the ioctls — the fault path fills them on demand.
	src := &prefillStream{
		size: pages * PageSize,
		runs: []prefillFakeRun{
			{offset: 0, end: 2 * PageSize, kind: sparse.Data, fill: 0xa5},
			{offset: 2 * PageSize, end: 4 * PageSize, kind: sparse.Zero},
			{offset: 4 * PageSize, end: 6 * PageSize, kind: sparse.Data, fill: 0x5d},
			{offset: 6 * PageSize, end: 7 * PageSize, kind: sparse.Hole},
			{offset: 7 * PageSize, end: 8 * PageSize, kind: sparse.Data, fill: 0x11},
		},
	}
	ops := newFakeIoctls()
	h, _ := loadHandler(t, pages, pages, prefillReader(t, src, pages*PageSize), ops.ops())

	st, err := h.Prefill(context.Background(), prefillReader(t, src, pages*PageSize))
	if err != nil {
		t.Fatal(err)
	}
	if st.Extents != 3 || st.PagesCopied != 5 || st.Skipped != 0 {
		t.Fatalf("stats = %+v, want extents=3 pages=5 skipped=0", st)
	}
	calls := ops.snapshot()
	if len(calls) != 3 {
		t.Fatalf("copies=%d want 3 (one per data extent)", len(calls))
	}
	wantData := []struct {
		page uint64
		fill byte
	}{{0, 0xa5}, {4, 0x5d}, {7, 0x11}}
	for i, c := range calls {
		if c.dst != unitCHVA+wantData[i].page*PageSize {
			t.Fatalf("copy %d dst=0x%x want 0x%x", i, c.dst, unitCHVA+wantData[i].page*PageSize)
		}
		if !bytes.Equal(c.data, bytes.Repeat([]byte{wantData[i].fill}, int(c.length))) {
			t.Fatalf("copy %d payload mismatch", i)
		}
	}
	for _, idx := range []uint64{0, 1, 4, 5, 7} {
		if h.state.Get(idx) != StateLoaded {
			t.Fatalf("page %d not Loaded", idx)
		}
	}
	for _, idx := range []uint64{2, 3, 6} {
		if h.state.Get(idx) != StateAbsent {
			t.Fatalf("page %d must stay Absent (zero/hole left to fault path)", idx)
		}
	}
	if got := h.Stats()["prefill_pages"]; got != 5 {
		t.Fatalf("prefill_pages = %d, want 5", got)
	}
}

func TestPrefillGuardsAndPartialResidency(t *testing.T) {
	const pages = 4
	src := &prefillStream{
		size: pages * PageSize,
		runs: []prefillFakeRun{
			{offset: 0, end: pages * PageSize, kind: sparse.Data, fill: 0xa5},
		},
	}
	reader := prefillReader(t, src, pages*PageSize)

	h, _ := loadHandler(t, pages, pages, reader, newFakeIoctls().ops())
	if _, err := h.Prefill(context.Background(), nil); err == nil {
		t.Fatal("nil source must be rejected")
	}

	ops := newFakeIoctls()
	h, _ = loadHandler(t, pages, pages, reader, ops.ops())
	// Pre-install page 0 via the mandatory path. The fake COPY succeeds for
	// the whole chunk, so the committed count must exclude the resident page.
	if err := h.EnsureLoaded(context.Background(), 0, PageSize); err != nil {
		t.Fatal(err)
	}
	st, err := h.Prefill(context.Background(), reader)
	if err != nil {
		t.Fatal(err)
	}
	if st.PagesCopied != pages-1 {
		t.Fatalf("pages=%d want %d (resident page not committed again)", st.PagesCopied, pages-1)
	}
}

func TestPrefillSurvivesRunAndCopyFailures(t *testing.T) {
	const pages = 4
	base := prefillStream{
		size: pages * PageSize,
		runs: []prefillFakeRun{
			{offset: 0, end: 2 * PageSize, kind: sparse.Data, fill: 0xa5},
			{offset: 2 * PageSize, end: 4 * PageSize, kind: sparse.Data, fill: 0x5d},
		},
	}

	// RunAt failure on the first extent: it is skipped, the second still
	// installs, and the pass succeeds (best-effort).
	fs := &failingStream{prefillStream: base, failOffset: 0, failErr: errors.New("boom")}
	ops := newFakeIoctls()
	h, _ := loadHandler(t, pages, pages, prefillReader(t, fs, pages*PageSize), ops.ops())
	st, err := h.Prefill(context.Background(), prefillReader(t, fs, pages*PageSize))
	if err != nil {
		t.Fatal(err)
	}
	if st.Skipped != 1 || st.PagesCopied != 3 || st.Extents != 2 {
		t.Fatalf("stats = %+v, want skipped=1 pages=3 extents=2 (cursor+page lands on extent1 remainder: page 1, then extent2: pages 2-3)", st)
	}

	// Copy EEXIST on every page: the pass steps through without failing.
	ops2 := newFakeIoctls()
	ops2.copyHook = func(int, uint64, []byte) (int64, error) { return 0, unix.EEXIST }
	h, _ = loadHandler(t, pages, pages, prefillReader(t, &base, pages*PageSize), ops2.ops())
	st, err = h.Prefill(context.Background(), prefillReader(t, &base, pages*PageSize))
	if err != nil {
		t.Fatal(err)
	}
	if st.PagesCopied != 0 || st.Extents != 2 {
		t.Fatalf("stats = %+v, want pages=0 extents=2", st)
	}
}

func TestPrefillConsecutiveRunFailuresBailOut(t *testing.T) {
	const pages = 4
	h, _ := loadHandler(t, pages, pages, prefillReader(t, &prefillStream{size: pages * PageSize}, pages*PageSize), newFakeIoctls().ops())
	st, err := h.Prefill(context.Background(), alwaysFailSource{})
	if err == nil {
		t.Fatalf("expected bail-out error, got stats %+v", st)
	}
	if st.Extents != 0 {
		t.Fatalf("no extent may be attempted after bail-out: %+v", st)
	}
}

func TestPrefillStopsAtByteBudget(t *testing.T) {
	const pages = 8
	// Four 1-page data extents; a 3-page budget must end the walk after the
	// third extent completes, leaving the fourth to the fault path.
	src := &prefillStream{
		size: pages * PageSize,
		runs: []prefillFakeRun{
			{offset: 0, end: 1 * PageSize, kind: sparse.Data, fill: 0x01},
			{offset: 1 * PageSize, end: 2 * PageSize, kind: sparse.Data, fill: 0x02},
			{offset: 2 * PageSize, end: 3 * PageSize, kind: sparse.Data, fill: 0x03},
			{offset: 3 * PageSize, end: 4 * PageSize, kind: sparse.Data, fill: 0x04},
		},
	}

	saved := prefillBudgetBytes
	t.Cleanup(func() { prefillBudgetBytes = saved })

	prefillBudgetBytes = 3 * PageSize
	ops := newFakeIoctls()
	h, _ := loadHandler(t, pages, pages, prefillReader(t, src, pages*PageSize), ops.ops())
	st, err := h.Prefill(context.Background(), prefillReader(t, src, pages*PageSize))
	if err != nil {
		t.Fatal(err)
	}
	if !st.Capped || st.Extents != 3 || st.PagesCopied != 3 {
		t.Fatalf("stats = %+v, want capped=true extents=3 pages=3", st)
	}
	if calls := ops.snapshot(); len(calls) != 3 {
		t.Fatalf("copies=%d want 3 (extent 4 must stay on the fault path)", len(calls))
	}

	// A budget above the layer size must neither truncate nor flag.
	prefillBudgetBytes = saved
	ops2 := newFakeIoctls()
	h, _ = loadHandler(t, pages, pages, prefillReader(t, src, pages*PageSize), ops2.ops())
	st, err = h.Prefill(context.Background(), prefillReader(t, src, pages*PageSize))
	if err != nil {
		t.Fatal(err)
	}
	if st.Capped || st.Extents != 4 || st.PagesCopied != 4 {
		t.Fatalf("stats = %+v, want capped=false extents=4 pages=4", st)
	}
}
