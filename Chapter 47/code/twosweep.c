$S$ = $\left\lceil \frac{\qTwoLstMathCf{light\_list\_end} - \qTwoLstMathCf{light\_list\_begin}}{\qTwoLstMathCf{MAX\_SUBSET\_LEN}} \right\rceil$
subset_stride = $S$
subset_offset = $\lfloor\xi_1$ * $S\rfloor$
$\xi_1$ = $\xi_1$ * $S$ - $\lfloor\xi_1$ * $S\rfloor$

total_weights = 0
float is_weights[MAX_SUBSET_LEN]

light_idx = light_list_begin + subset_offset
for (i = 0; i < MAX_SUBSET_LEN; ++i) {
	if (light_idx >= light_list_end) {
		break
	}
	w = light_contrib(v, p, n, light_pointers[light_idx])
	is_weights[i] = w
	total_weights += w
	
	light_idx += subset_stride
}

$\xi_1$ *= total_weights
mass = 0

light_idx = light_list_begin + subset_offset
for (i = 0; i < MAX_SUBSET_LEN; ++i) {
	if (light_idx >= light_list_end) {
		break
	}	
	mass = is_weights[i]
	
	$\xi_1$ -= mass
	if not ($\xi_1$ > 0) {
		break
	}
	light_idx += subset_stride
}

probability = mass / (total_weights * $S$)
return (light_pointers[light_idx], probability)