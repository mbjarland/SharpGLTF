using System;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpGLTF.Schema2
{
    /// <summary>
    /// Somewhat optimized version of finding min/max values in a vector of floats. Please note some effort
    /// has been made to test a multithreaded version of this as well, but it was not faster than this implementation
    /// for the data sets it was tested against. If anybody feels so inclined, please feel free to try and improve
    /// this further.
    /// </summary>
    public static class VectorMinMax
    {
        public static (float[] min, float[] max) FindMinMax(ReadOnlySpan<float> data, int dimensions)
        {
            if (data.Length % dimensions != 0)
                throw new ArgumentException($"Data length must be divisible by {dimensions}");

            var min = new float[dimensions];
            var max = new float[dimensions];
            Array.Fill(min, float.MaxValue);
            Array.Fill(max, float.MinValue);


            if (dimensions == 3 && data.Length >= 24)
            {
                // Special optimized path for 3D vectors
                ProcessSIMD3D(data, min, max);
            } else
            {
                // General case for other dimensions
                ProcessSIMD(data, dimensions, min, max);
            }

            return (min, max);
        }

        // ReSharper disable once InconsistentNaming
        private static unsafe void ProcessSIMD(ReadOnlySpan<float> data, int dimensions, float[] min, float[] max)
        {
            fixed (float* ptr = data)
            {
                if (Avx2.IsSupported && data.Length >= dimensions * 8)
                {
                    // intel processors, 8 floats = 256 bits
                    ProcessWithAVX(ptr, data.Length, dimensions, min, max);
                } else if (Vector.IsHardwareAccelerated && data.Length >= dimensions * Vector<float>.Count)
                {
                    // on arm / apple silicon etc, Vector<float>.Count usually == 4. 4 floats = 128 bits
                    ProcessWithVector(ptr, data.Length, dimensions, min, max);
                } else
                {
                    // and otherwise fall back to for loops and scalar operations, comparing one float at a time
                    ProcessScalar(ptr, data.Length, dimensions, min, max);
                }
            }
        }

        // ReSharper disable once InconsistentNaming
        private static unsafe void ProcessWithAVX(float* ptr, int length, int dimensions, float[] min, float[] max)
        {
            var minVecs = new Vector256<float>[dimensions];
            var maxVecs = new Vector256<float>[dimensions];

            for (int d = 0; d < dimensions; d++)
            {
                minVecs[d] = Vector256.Create(float.MaxValue);
                maxVecs[d] = Vector256.Create(float.MinValue);
            }

            int i                = 0;
            int vectorizedLength = length - (length % (dimensions * 8));

            for (; i < vectorizedLength; i += dimensions * 8)
            {
                for (int d = 0; d < dimensions; d++)
                {
                    var vec = Avx.LoadVector256(ptr + i + d * 8);
                    minVecs[d] = Avx.Min(minVecs[d], vec);
                    maxVecs[d] = Avx.Max(maxVecs[d], vec);
                }
            }

            var temp = stackalloc float[8];
            for (int d = 0; d < dimensions; d++)
            {
                Avx.Store(temp, minVecs[d]);
                for (int j = 0; j < 8; j++)
                {
                    min[d] = Math.Min(min[d], temp[j]);
                }

                Avx.Store(temp, maxVecs[d]);
                for (int j = 0; j < 8; j++)
                {
                    max[d] = Math.Max(max[d], temp[j]);
                }
            }

            ProcessRemainingElements(ptr, i, length, dimensions, min, max);
        }

        private static unsafe void ProcessWithVector(float* ptr, int length, int dimensions, float[] min, float[] max)
        {
            var minVecs    = new Vector<float>[dimensions];
            var maxVecs    = new Vector<float>[dimensions];
            int vectorSize = Vector<float>.Count;

            for (int d = 0; d < dimensions; d++)
            {
                minVecs[d] = new Vector<float>(float.MaxValue);
                maxVecs[d] = new Vector<float>(float.MinValue);
            }

            int i                = 0;
            int vectorizedLength = length - (length % (dimensions * vectorSize));

            // Main vectorized loop
            for (; i < vectorizedLength; i += dimensions * vectorSize)
            {
                for (int d = 0; d < dimensions; d++)
                {
                    var span = new ReadOnlySpan<float>(ptr + i + d * vectorSize, vectorSize);
                    var vec  = new Vector<float>(span);
                    minVecs[d] = Vector.Min(minVecs[d], vec);
                    maxVecs[d] = Vector.Max(maxVecs[d], vec);
                }
            }

            // Reduce vectors to scalar values
            for (int d = 0; d < dimensions; d++)
            {
                min[d] = float.MaxValue;
                max[d] = float.MinValue;

                for (int j = 0; j < vectorSize; j++)
                {
                    min[d] = Math.Min(min[d], minVecs[d][j]);
                    max[d] = Math.Max(max[d], maxVecs[d][j]);
                }
            }

            ProcessRemainingElements(ptr, i, length, dimensions, min, max);
        }

        // ReSharper disable once InconsistentNaming
        private static unsafe void ProcessSIMD3D(ReadOnlySpan<float> data, float[] min, float[] max)
        {
            fixed (float* ptr = data)
            {
                if (Avx2.IsSupported && data.Length >= 24)
                {
                    ProcessWithAVX3D(ptr, data.Length, min, max);
                } else if (Vector.IsHardwareAccelerated && data.Length >= 12)
                {
                    ProcessWithVector3D(ptr, data.Length, min, max);
                } else
                {
                    ProcessScalar(ptr, data.Length, 3, min, max);
                }
            }
        }


        // ReSharper disable once InconsistentNaming
        private static unsafe void ProcessWithAVX3D(float* ptr, int length, float[] min, float[] max)
        {
            // Initialize vectors for each dimension
            var min0 = Vector256.Create(float.MaxValue);
            var min1 = Vector256.Create(float.MaxValue);
            var min2 = Vector256.Create(float.MaxValue);

            var max0 = Vector256.Create(float.MinValue);
            var max1 = Vector256.Create(float.MinValue);
            var max2 = Vector256.Create(float.MinValue);

            int i                = 0;
            int vectorizedLength = length - (length % 24); // Process in chunks of 24 floats (8 vectors × 3 dimensions)

            // Main processing loop - handles 8 vectors at a time
            for (; i < vectorizedLength; i += 24)
            {
                var c0 = Avx.LoadVector256(ptr + i);
                min0 = Avx.Min(min0, c0);
                max0 = Avx.Max(max0, c0);

                var c1 = Avx.LoadVector256(ptr + i + 8);
                min1 = Avx.Min(min1, c1);
                max1 = Avx.Max(max1, c1);

                var c2 = Avx.LoadVector256(ptr + i + 16);
                min2 = Avx.Min(min2, c2);
                max2 = Avx.Max(max2, c2);
            }

            // Reduce the vectors to scalar values
            var temp = stackalloc float[8];

            // Process min values
            Avx.Store(temp, min0);
            min[0] = temp[0];
            for (int j = 1; j < 8; j++) min[0] = Math.Min(min[0], temp[j]);

            Avx.Store(temp, min1);
            min[1] = temp[0];
            for (int j = 1; j < 8; j++) min[1] = Math.Min(min[1], temp[j]);

            Avx.Store(temp, min2);
            min[2] = temp[0];
            for (int j = 1; j < 8; j++) min[2] = Math.Min(min[2], temp[j]);

            // Process max values
            Avx.Store(temp, max0);
            max[0] = temp[0];
            for (int j = 1; j < 8; j++) max[0] = Math.Max(max[0], temp[j]);

            Avx.Store(temp, max1);
            max[1] = temp[0];
            for (int j = 1; j < 8; j++) max[1] = Math.Max(max[1], temp[j]);

            Avx.Store(temp, max2);
            max[2] = temp[0];
            for (int j = 1; j < 8; j++) max[2] = Math.Max(max[2], temp[j]);

            // Process remaining elements
            ProcessRemainingElements(ptr, i, length, 3, min, max);
        }

        private static unsafe void ProcessWithVector3D(float* ptr, int length, float[] min, float[] max)
        {
            int vectorSize = Vector<float>.Count;

            // Initialize vectors for each dimension
            var min0 = new Vector<float>(float.MaxValue);
            var min1 = new Vector<float>(float.MaxValue);
            var min2 = new Vector<float>(float.MaxValue);

            var max0 = new Vector<float>(float.MinValue);
            var max1 = new Vector<float>(float.MinValue);
            var max2 = new Vector<float>(float.MinValue);

            int i                = 0;
            int vectorizedLength = length - (length % (3 * vectorSize));

            // Main processing loop
            for (; i < vectorizedLength; i += 3 * vectorSize)
            {
                var vec0 = new Vector<float>(new ReadOnlySpan<float>(ptr + i, vectorSize));
                min0 = Vector.Min(min0, vec0);
                max0 = Vector.Max(max0, vec0);

                var vec1 = new Vector<float>(new ReadOnlySpan<float>(ptr + i + vectorSize, vectorSize));
                min1 = Vector.Min(min1, vec1);
                max1 = Vector.Max(max1, vec1);

                var vec2 = new Vector<float>(new ReadOnlySpan<float>(ptr + i + 2 * vectorSize, vectorSize));
                min2 = Vector.Min(min2, vec2);
                max2 = Vector.Max(max2, vec2);
            }

            // Reduce vectors to scalar values
            min[0] = float.MaxValue;
            min[1] = float.MaxValue;
            min[2] = float.MaxValue;
            max[0] = float.MinValue;
            max[1] = float.MinValue;
            max[2] = float.MinValue;

            for (int j = 0; j < Vector<float>.Count; j++)
            {
                min[0] = Math.Min(min[0], min0[j]);
                min[1] = Math.Min(min[1], min1[j]);
                min[2] = Math.Min(min[2], min2[j]);

                max[0] = Math.Max(max[0], max0[j]);
                max[1] = Math.Max(max[1], max1[j]);
                max[2] = Math.Max(max[2], max2[j]);
            }

            // Process remaining elements
            ProcessRemainingElements(ptr, i, length, 3, min, max);
        }

        private static unsafe void ProcessScalar(float* ptr, int length, int dimensions, float[] min, float[] max)
        {
            for (int i = 0; i < length; i += dimensions)
            {
                for (int d = 0; d < dimensions; d++)
                {
                    min[d] = Math.Min(min[d], ptr[i + d]);
                    max[d] = Math.Max(max[d], ptr[i + d]);
                }
            }
        }

        private static unsafe void ProcessRemainingElements(float* ptr, int start, int length, int dimensions, float[] min, float[] max)
        {
            for (int i = start; i < length; i += dimensions)
            {
                for (int d = 0; d < dimensions; d++)
                {
                    min[d] = Math.Min(min[d], ptr[i + d]);
                    max[d] = Math.Max(max[d], ptr[i + d]);
                }
            }
        }
    }
}