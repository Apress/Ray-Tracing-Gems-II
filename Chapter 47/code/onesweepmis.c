$S$ = $\left\lceil \frac{\qTwoLstMathCf{light\_list\_end} - \qTwoLstMathCf{light\_list\_begin}}{\qTwoLstMathCf{MAX\_SUBSET\_LEN}} \right\rceil$
subset_stride = $S$

selected = (light_idx: -1, mass: 0)
total_weights = 0

// Additional information and randomization for MIS
hit_caught = not hit_emitter
other_weights = 0 // Weights excluding hit_emitter stride
pending_weight = 0
FastRng small_rnd(seed: $\xi_1$, mod: $S$)

light_offset = light_list_begin
for (i = 0; i < MAX_SUBSET_LEN; ++i) {
	light_idx = light_offset + small_rnd()
	if (light_idx >= light_list_end) {
		// Detect if hit_emitter is in current stride.
		hit_caught ||=   light_offset < light_list_end
		              && light_pointers[light_offset] <= hit_emitter
		break
	}
	w = light_contrib(v, p, n, light_pointers[light_idx])
	// Accumulate all weights outside hit_emitter's stride.
	wo = w
	if (not hit_caught && hit_emitter <= light_pointers[light_idx]) {
		// Is the emitter in this or the last stride?
		if (light_pointers[light_offset] <= hit_emitter)
			wo = 0 // This stride
		else
			pending_weight = 0 // Last stride
		hit_caught = true // Found hit_emitter
	}
	other_weights += pending_weight
	pending_weight = wo
	
	if (w > 0) {
		$\tau$ = $\frac{ \qTwoLstMathCf{total\_weights} }{\qTwoLstMathCf{total\_weights} + \qTwoLstMathCf{w}}$; total_weights += w
		if ($\xi_1 < \tau$) { $\xi_1$ /= $\tau$ }
		else { selected = (light_pointers[light_idx], w); $\xi_1$ =  $\frac{\xi_1 - \tau}{1 - \tau}$ }
		$\xi_1$ = clamp($\xi_1$, 0, MAX_BELOW_ONE)
	}
	light_offset += subset_stride
}
// Compute pseudo-marginal probability of sampling hit_emitter.
if (hit_caught)
	other_weights += pending_weight
hit_w = light_contrib(v, p, n, hit_emitter)
hit_probability = hit_w / ((other_weights + hit_w) * $S$)

probability = selected.mass / (total_weights * $S$)
return (selected.light, probability, hit_probability)
