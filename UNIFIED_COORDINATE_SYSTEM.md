For subpixel coordinates, f32 is not a good fit. For bigger numbers, the grantularity/accuracy decreases, which is a bad.
I want you to output all coordinates in `euclid::Point2D<fixed::types::U20F12>` instead. To be able to represent the very top left subpixel of the top-left pixel with unsigned numbers, the top left pixel-center has to be 0.5/0.5. 

Only public coordinate inputs/outputs become fixed-point. If floats are expected in public coordinate parameters, use fixed::ToFixed, which should still compile for API users which provide f32 today (compile-compat only: the f32 is taken as the raw value in the new convention, no +0.5 offset). Angles, scales, scores, thresholds and gradient/Sobel math stay f32. Internal integer representations (MatchRaw, Feature pos, tl) stay as-is; convert at the API boundary only.

Add a non-pub fn to_image_coord() helper, which converts uint to float types, adding this 0.5 offset. Use a internal trait ToImageCoord as a argument and implement it for `u8, u16, u32, u64, usize` and also `euclid::Point2D<T>` if T: ToImageCoord.

Execution plan: Yield control after each step
1.) Make sure that the output of `debug_visual` stays exactly the same. Zero differences in the output is a absolute must. Create a test which verifies this, before you start, that the sha2 (add to dev-dependencies) of all pixel values remains the same (first detect it, then store the hash inline right next to the test). Cover both the detector and scale test scenarios... 
2.) Change the f32 of all input/outputs coordinates (not angles!) to the new coordinate system convention (+0.5), keep debug_visual the same...
3.) Delete the previously added debug_visual test, remove the sha2 dev-dependency again and replace the public coordinate f32 values (coordinates only, not angles!) with fixed::types::U20F12
