"""Sim-metrics -> ranked bottleneck -> prescribed intrinsic/mechanism fix.

Reads only fields evaluate() already emits (roofline attach_roofline + anti-cheat
+ PMU). The memory-vs-compute REGIME is structural (AI/bw/peak) and reliable; the
DISTANCE to the ceiling (eta) is timing and box-gated. Pure: no I/O, no heavy imports."""
from dataclasses import dataclass

from hexkernels.gym import reward as R

_ERR_MAX = 600


@dataclass
class Prescription:
    bottleneck: str
    fix_text: str
    target: str
    reliability: str


_MISSING = {
    "dma": ("correct, but used_dma=false -- your load compiled to a blocking copy, not the DMA "
            "engine; stage the tile through VTCM with a descriptor (dmstart/dmwait) and prefetch "
            "tile c+1 while computing tile c."),
    "vtcm": ("correct, but used_vtcm=false -- you never staged data into the VTCM scratchpad; "
             "allocate a VTCM tile and operate on it there."),
    "hvx": ("correct, but used_hvx_compute=false (no genuine HVX vector compute) -- vectorize the "
            "inner loop with HVX (vrmpy/vdmpy/vmpy); check element types, 128-B alignment, and the "
            "n%128 tail."),
    "hmx": ("correct, but used_hmx=false -- this matmul never reached the HMX engine; use the "
            "tile-matmul flow (crouton pack + mxclracc/mxmem), not a scalar/HVX MAC loop."),
    "l2fetch": ("correct, but used_l2fetch=false -- add an l2fetch prefetch of the next block to "
                "hide DDR latency."),
}

# Memory-bound-specific wording (regime-driven branch): only used when `_regime(feedback)=="memory"`,
# separate from `_MISSING["l2fetch"]` (the REQUIRED-but-missing branch) so the required-mechanism
# loop never falsely claims "memory-bound" for a task where l2fetch is merely required, not measured
# to be on the memory-bound path.
_MEMORY_BOUND_L2FETCH = ("memory-bound and used_l2fetch=false -- add an l2fetch prefetch of the "
                         "next block to hide DDR latency.")

# Priority order for reporting a missing REQUIRED mechanism (staging/compute engines first).
_PRIORITY = ["dma", "vtcm", "hmx", "hvx", "l2fetch"]


def _regime(feedback):
    ai, eff, ceil = feedback.get("AI"), feedback.get("eff_bw"), feedback.get("ceiling")
    if ai is None or eff is None or ceil is None:
        return None
    return "memory" if ceil >= ai * eff - 1e-9 else "compute"


def analyze(feedback, spec, on_box=False):
    if not feedback.get("compiled"):
        err = (feedback.get("error_text") or "").strip()[:_ERR_MAX]
        return [Prescription("build", "Your kernel DID NOT COMPILE:\n" + err, "compile", "reliable")]
    if not feedback.get("correct"):
        text = "Your kernel compiled but is INCORRECT -- fix the logic."
        if feedback.get("used_hvx_src") and not feedback.get("used_hvx_compute"):
            text += (" NOTE: your source uses HVX intrinsics but the compiled kernel is SCALAR -- "
                     "they did not vectorize (check element types, alignment, n%128 tail).")
        return [Prescription("logic", text, "correctness", "reliable")]

    out = []
    mechs = R.target_mechanisms(spec)
    for m in _PRIORITY:
        if m in mechs and not feedback.get(R.MECH_FLAG[m]):
            out.append(Prescription("missing_mechanism", _MISSING[m], m, "reliable"))
    if _regime(feedback) == "memory" and not feedback.get("used_l2fetch") and "l2fetch" not in mechs:
        out.append(Prescription("ddr_latency", _MEMORY_BOUND_L2FETCH, "l2fetch", "reliable"))
    if on_box and R.is_genuine(feedback, spec):
        eta = feedback.get("eta")
        if eta is not None and eta < 0.5:
            out.append(Prescription(
                "under_utilized",
                (f"correct + genuine but roofline efficiency eta={eta*100:.0f}% -- go deeper: "
                 "wider accumulation, fewer passes, tile to fit VTCM. (simulator-only)"),
                "optimize", "box_gated"))
    return out


def render(prescriptions, turn, max_turns):
    head = f"[turn {turn}/{max_turns}] "
    if not prescriptions:
        return head + "Your kernel is CORRECT and uses the target mechanism. Now reduce kernel cycles."
    return head + prescriptions[0].fix_text
