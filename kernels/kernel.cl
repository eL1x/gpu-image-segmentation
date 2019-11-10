// Set sampler
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST 
    | CLK_ADDRESS_CLAMP_TO_EDGE;


// Kernel to assign initial label values for pixels
__kernel void init_labels(__read_only image2d_t image, 
        __write_only image2d_t labels_array)
{
	// Get x, y coordinates in image and image width
	const int x = get_global_id(0);
	const int y = get_global_id(1);
    const int width = get_global_size(0);
	
    const float pixelVal = read_imagef(image, sampler, (int2)(x, y)).s0;

    // If pixel value is 1 assign label according to its position
    float labelValue = 0;
    if (pixelVal == 1.0)
        labelValue = y * width + x + 1;
	
	// Write pixel label to labels array
    write_imageui(labels_array, (int2)(x, y), convert_uint(labelValue));
}


// Kernel to find minimal positive label value over a neighborhood
__kernel void neigh_min_label(__read_only image2d_t labels_in, 
        __write_only image2d_t labels_out)
{
	// Init variables
	const int x = get_global_id(0);
	const int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int maskSize = 3;
	
    // Find minimal positive label only for labels different from zero
    float currentPixel = convert_float4(read_imageui(labels_in, sampler, 
        (int2)(x, y))).x;
    if (currentPixel != 0.0)
    {
        // Get minimal label value in mask
        float minLabel = (width * height + 1);
        int padding = (int)(floor(maskSize / 2.0));
        for (int i = -padding; i <= padding; ++i)
        {
            for (int j = -padding; j <= padding; ++j)
            {
                if (x + i >= 0 && x + i <= (width - 1) && y + j >= 0 
                        && y + j <= height - 1)
                {
                    float label = convert_float4(read_imageui(labels_in, 
                        sampler, (int2)(x + i, y + j))).x;

                    if (label != 0)
                        minLabel = min(minLabel, label);
                    // minLabel = 2;
                }
            }		
        }	
        // Write new label to out matrix
        write_imageui(labels_out, (int2)(x, y), convert_uint(minLabel));
    }
}


// Kernel to find minimal positive label value over a neighborhood
__kernel void compare_matrices(__read_only image2d_t matrix1, 
        __read_only image2d_t matrix2, __global float *output)
{
	// Init variables
	const int x = get_global_id(0);
	const int y = get_global_id(1);
    const int width = get_global_size(0);

    // Get elements to compare
    float element1 = convert_float4(read_imageui(matrix1, sampler, 
        (int2)(x, y))).x;
    float element2 = convert_float4(read_imageui(matrix2, sampler, 
        (int2)(x, y))).x;

    // Compare them
    float outValue = 0;
    if (element1 != element2)
        outValue = 1;

    // Write to Buffer
    int idx = y * width + x;
    output[idx] = outValue;
}