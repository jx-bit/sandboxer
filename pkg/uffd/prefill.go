package uffd

import (
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/kuasar-sandbox/accelerator/pkg/sparse"
	"github.com/kuasar-sandbox/sandboxer/internal/readretry"
	"golang.org/x/sys/unix"
)

// prefillReadChunk bounds one staging read and its multi-page UFFDIO_COPY
// during restore prefill. Current manifests hold chunks no larger than 1 MiB,
// so one iteration covers a typical self-layer extent; larger contiguous runs
// are still served correctly through the staged innerOffset loop.
const prefillReadChunk = 1 << 20

// prefillBudgetBytes caps how much source data one prefill pass may read.
// The self checkpoint layer of a PAUSED sandbox is the resume hot set and
// measures ~3-4 MiB in the reference workload, so the budget never truncates
// it. A TEMPLATE snapshot restore, however, exposes the template's whole
// capture as its self layer (23k pages / 95 MB observed) — reading all of it
// serially (~125 ms) costs more than the first-dial storm it would absorb
// (~93 ms), turning create into a net regression. The budget keeps the pass
// profitable for any layer size: truncated passes absorb the leading pages of
// the storm and leave the rest to the fault path (AB v1 finding, 2026-11-20).
// A var so tests can shrink it.
var prefillBudgetBytes = uint64(8) << 20

// PrefillStats summarizes one best-effort prefill pass.
type PrefillStats struct {
	Extents     uint64 // Data runs attempted
	PagesCopied uint64 // pages committed Absent→Loaded by this pass
	BytesRead   uint64
	ReadNs      uint64
	CopyNs      uint64
	Skipped     uint64 // extents abandoned (read/install failure)
	Capped      bool   // stopped at prefillBudgetBytes before the layer ended
}

// Prefill batch-installs the data extents of src into the memfd while the VM
// is Paused (restore PostSpawn: after CH api_ready, before vm.resume). The
// caller passes a SnapshotReader bound to the self checkpoint layer — the
// bundle stores exactly the Dirty∩Materialized pages captured at pause time,
// which is the resume first-touch set — so installing them ahead of the vCPUs
// removes the serialized per-window cache RPC round trips that dominate the
// first-dial fault storm (docs/optimizations/p3-uffd-prefill.md §2;
// Eulogizethesun/kuasar-sandbox#5 方案 A).
//
// Zero and hole runs are deliberately left to the fault path (ZEROPAGE is
// near-free); only Data runs are read and installed. The pass reads at most
// prefillBudgetBytes of source data: a paused sandbox's self layer (the real
// hot set) fits far below it, while a template snapshot's self layer can
// dwarf the storm it would absorb — the budget truncates such layers instead
// of paying more than the fault path would.
//
// Best-effort: enumeration
// and read/install failures skip forward and surface in PrefillStats.Skipped —
// prefill must never fail the restore. The caller holds no guest waiters while
// the VM is Paused, so no UFFDIO_WAKE is issued; an EEXIST (a folio installed
// by another actor — not expected pre-resume) steps past that folio and keeps
// filling the extent, and the fault path converges anything left after resume.
func (h *Handler) Prefill(caller context.Context, src SnapshotReader) (PrefillStats, error) {
	started := time.Now()
	st, err := h.prefill(caller, src)
	h.stats.prefillExtents.Add(st.Extents)
	h.stats.prefillPages.Add(st.PagesCopied)
	h.stats.prefillNs.Add(uint64(time.Since(started).Nanoseconds()))
	return st, err
}

func (h *Handler) prefill(caller context.Context, src SnapshotReader) (PrefillStats, error) {
	if src == nil {
		return PrefillStats{}, errors.New("uffd: prefill: nil source")
	}
	h.uffdsMu.Lock()
	if h.closing.Load() || h.ctx.Err() != nil || !h.started {
		h.uffdsMu.Unlock()
		return PrefillStats{}, fmt.Errorf("uffd: handler is not running")
	}
	h.users.Add(1) // serialized with closing and Wait, same admission as EnsureLoaded
	h.uffdsMu.Unlock()
	defer h.users.Done()

	ctx, cancel := context.WithCancel(caller)
	stop := context.AfterFunc(h.ctx, cancel)
	defer func() { stop(); cancel() }()

	var st PrefillStats
	staging := make([]byte, prefillReadChunk)
	// The unbounded limit keeps RunAt bounded by the reader's own RAM size
	// (StreamSnapshotSource clamps to ramSize and returns io.EOF past it).
	const walkLimit = ^uint64(0)
	const maxConsecutiveRunFailures = 1024
	consecutive := 0
	for cursor := uint64(0); ; {
		if ctx.Err() != nil {
			return st, ctx.Err()
		}
		run, err := src.RunAt(cursor, walkLimit)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return st, nil
			}
			st.Skipped++
			consecutive++
			if consecutive > maxConsecutiveRunFailures {
				return st, fmt.Errorf("uffd: prefill: reader failed %d consecutive runs at 0x%x", consecutive, cursor)
			}
			cursor += PageSize // unresolvable extent: step one page and keep walking
			continue
		}
		consecutive = 0
		if run.End() <= run.Offset() || run.Offset() != cursor {
			return st, fmt.Errorf("uffd: prefill: reader returned misaligned run [0x%x,0x%x) at 0x%x",
				run.Offset(), run.End(), cursor)
		}
		if run.Kind() != sparse.Data {
			cursor = run.End() // zero/hole: the fault path fills these on demand
			continue
		}
		st.Extents++
		if err := h.prefillExtent(ctx, run, staging, &st); err != nil {
			if ctx.Err() != nil {
				return st, ctx.Err()
			}
			st.Skipped++
			if h.logf != nil {
				h.logf("uffd: prefill: extent [0x%x,0x%x) skipped: %v", run.Offset(), run.End(), err)
			}
		}
		cursor = run.End()
		if st.BytesRead >= prefillBudgetBytes {
			// Budget spent: further reads would cost more than the fault
			// path they shortcut (template-size layers). The pages already
			// installed stay; the fault path converges the remainder.
			st.Capped = true
			return st, nil
		}
	}
}

// prefillExtent stages one Data run through the shared buffer and installs
// each page-aligned chunk. Source reads run outside installMu (its contract
// forbids source I/O under the lock); each install validates ownership and
// conditionally commits state under installMu, matching loadPage's rules.
func (h *Handler) prefillExtent(ctx context.Context, run sparse.Run, staging []byte, st *PrefillStats) error {
	for off := run.Offset(); off < run.End(); {
		n := uint64(len(staging))
		if n > run.End()-off {
			n = run.End() - off
		}
		n = n / PageSize * PageSize // COPY must be page-aligned
		if n == 0 {
			// Sub-page tail cannot exist for snapshot runs (they are
			// page-bounded by construction); treat as fully consumed.
			return nil
		}
		readStarted := time.Now()
		if _, err := readretry.ReadAt(ctx, int(n), func() (int, error) {
			return run.ReadAt(ctx, staging[:n], off-run.Offset())
		}); err != nil {
			st.ReadNs += uint64(time.Since(readStarted).Nanoseconds())
			return err
		}
		st.ReadNs += uint64(time.Since(readStarted).Nanoseconds())
		st.BytesRead += n
		processed, err := h.prefillInstall(off, staging[:n], st)
		switch {
		case err != nil && errors.Is(err, unix.EEXIST):
			// A folio is already present (not expected while Paused). The
			// copied prefix (if any) is committed; step past the present
			// folio and keep filling this extent.
			off += processed + PageSize
		case err != nil:
			return err
		case processed == 0:
			// Chunk clipped to zero by a region boundary (x86 low/high
			// split): step one page; the boundary tail stays on the fault
			// path, same as the urgent path.
			off += PageSize
		default:
			off += processed
		}
	}
	return nil
}

// prefillInstall runs under installMu: it clips [off,+len) to the owning CH
// registration, batch-COPYs the page-aligned clip, and conditionally commits
// Absent→Loaded for the completed prefix. It returns the page-aligned byte
// count consumed so the caller makes forward progress across region
// boundaries. EEXIST is reported with the copied prefix; the caller decides
// whether to step past it or abandon the extent.
func (h *Handler) prefillInstall(off uint64, data []byte, st *PrefillStats) (uint64, error) {
	h.installMu.Lock()
	defer h.installMu.Unlock()
	if err := h.ctx.Err(); err != nil {
		return 0, err
	}
	va, fd, covered, err := h.addrMap.registration(off, uint64(len(data)))
	if err != nil {
		return 0, err
	}
	if fd <= 0 {
		return 0, fmt.Errorf("uffd: prefill: no adopted CH registration at 0x%x", off)
	}
	if covered < uint64(len(data)) {
		covered = covered / PageSize * PageSize
		data = data[:covered]
	}
	pageIdx := off / PageSize
	pages := uint64(len(data)) / PageSize
	if pages == 0 {
		return 0, nil // everything fell outside the owning region
	}
	if h.state.RunLength(pageIdx, pages, StateLoaded) == pages {
		return uint64(len(data)), nil // fully resident; nothing to install
	}
	copyStarted := time.Now()
	completed, copyErr := h.copyCleanLocked(fd, va, data)
	st.CopyNs += uint64(time.Since(copyStarted).Nanoseconds())
	done, err := checkedCompletion(completed, uint64(len(data)))
	if err != nil {
		return 0, err
	}
	donePages := done / PageSize
	switch {
	case copyErr == nil:
		// Deliberately NOT counted in pagesCopied/copy_calls: those stay
		// fault-path-only so the restore.uffd_delta snapshot keeps its
		// baseline meaning (prefilled pages must show up as a pages_copied
		// DROP, with the batch accounting in the prefill_* namespace).
		committed := h.state.SetRangeIf(pageIdx, pageIdx+donePages, StateAbsent, StateLoaded)
		st.PagesCopied += committed
		return uint64(len(data)), nil
	case errors.Is(copyErr, unix.EEXIST):
		// Converge state for the copied prefix; the present folio keeps its
		// owner's bytes (COPY never overwrites an installed folio).
		h.state.SetRangeIf(pageIdx, pageIdx+donePages, StateAbsent, StateLoaded)
		return done, copyErr
	default:
		return done, copyErr
	}
}
