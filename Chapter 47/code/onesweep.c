$S$ = $\left\lceil \frac{\qTwoLstMathCf{light\_list\_end} - \qTwoLstMathCf{light\_list\_begin}}{\qTwoLstMathCf{MAX\_SUBSET\_LEN}} \right\rceil$
subset_stride = $S$
subset_offset = $\lfloor\xi_1$ * $S\rfloor$
$\xi_1$ = $\xi_1$ * $S$ - $\lfloor\xi_1$ * $S\rfloor$

selected = (light_idx: -1, mass: 0)
total_weights = 0

light_idx = light_list_begin + subset_offset
for (i = 0; i < MAX_SUBSET_LEN; ++i) {
	if (light_idx >= light_list_end) {
		break
	}
	w = light_contrib(v, p, n, light_pointers[light_idx])	
	if (w > 0) {
		$\tau$ = $\frac{ \qTwoLstMathCf{total\_weights} }{\qTwoLstMathCf{total\_weights} + \qTwoLstMathCf{w}}$
		total_weights += w
		
		if ($\xi_1 < \tau$) {
			$\xi_1$ /= $\tau$
		} else {
			selected = (light_pointers[light_idx], w)
			$\xi_1$ =  $\frac{\xi_1 - \tau}{1 - \tau}$
		}
		$\xi_1$ = clamp($\xi_1$, 0, MAX_BELOW_ONE) // Avoid numerical problems.
	}

	light_idx += subset_stride
}

probability = selected.mass / (total_weights * $S$)
return (selected.light, probability)
