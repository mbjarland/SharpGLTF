using System;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpGLTF.Schema2
{
    /// <summary>
    /// Somewhat optimized version of finding min/max values in a vector of floats. Please note some effort
    /// has been made to test a multi threaded version of this as well but it was not faster than this implementation
    /// for the data sets it was tested against. If anybody feels so inclined, please feel free to try and improve
    /// this further.
    /// </summary>
    public static class VectorMinMax
    {
        public static (float[] min, float[] max) FindMinMax(ReadOnlySpan<float> data, int dimensions) {
            if (data.Length % dimensions != 0)
                throw new ArgumentException($"Data length must be divisible by {dimensions}");

            var min = new float[dimensions];
            var max = new float[dimensions];
            Array.Fill(min, float.MaxValue);
            Array.Fill(max, float.MinValue);

            // Just use SIMD without parallelization for each individual call
            ProcessSIMD(data, dimensions, min, max);

            return (min, max);
        }

        // ReSharper disable once InconsistentNaming
        private static unsafe void ProcessSIMD(ReadOnlySpan<float> data, int dimensions, float[] min, float[] max) {
            fixed (float* ptr = data) {
                if (Avx2.IsSupported && data.Length >= dimensions * 8) {
                    // intel processors, 8 floats = 256 bits
                    ProcessWithAVX(ptr, data.Length, dimensions, min, max);
                } else if (Vector.IsHardwareAccelerated && data.Length >= dimensions * Vector<float>.Count) {
                    // on arm / apple silicon etc, Vector<float>.Count usually == 4. 4 floats = 128 bits
                    ProcessWithVector(ptr, data.Length, dimensions, min, max);
                } else {
                    // and otherwise fall back to for loops and scalar operations, comparing one float at a time
                    ProcessScalar(ptr, data.Length, dimensions, min, max);
                }
            }
        }

        // ReSharper disable once InconsistentNaming
        private static unsafe void ProcessWithAVX(float* ptr, int length, int dimensions, float[] min, float[] max) {
            var minVecs = new Vector256<float>[dimensions];
            var maxVecs = new Vector256<float>[dimensions];

            for (int d = 0; d < dimensions; d++) {
                minVecs[d] = Vector256.Create(float.MaxValue);
                maxVecs[d] = Vector256.Create(float.MinValue);
            }

            int i                = 0;
            int vectorizedLength = length - (length % (dimensions * 8));

            for (; i < vectorizedLength; i += dimensions * 8) {
                for (int d = 0; d < dimensions; d++) {
                    var vec = Avx.LoadVector256(ptr + i + d * 8);
                    minVecs[d] = Avx.Min(minVecs[d], vec);
                    maxVecs[d] = Avx.Max(maxVecs[d], vec);
                }
            }

            var temp = stackalloc float[8];
            for (int d = 0; d < dimensions; d++) {
                Avx.Store(temp, minVecs[d]);
                for (int j = 0; j < 8; j++) {
                    min[d] = Math.Min(min[d], temp[j]);
                }

                Avx.Store(temp, maxVecs[d]);
                for (int j = 0; j < 8; j++) {
                    max[d] = Math.Max(max[d], temp[j]);
                }
            }

            ProcessRemainingElements(ptr, i, length, dimensions, min, max);
        }

        private static unsafe void ProcessWithVector(float* ptr, int length, int dimensions, float[] min, float[] max) {
            var minVecs    = new Vector<float>[dimensions];
            var maxVecs    = new Vector<float>[dimensions];
            int vectorSize = Vector<float>.Count;

            for (int d = 0; d < dimensions; d++) {
                minVecs[d] = new Vector<float>(float.MaxValue);
                maxVecs[d] = new Vector<float>(float.MinValue);
            }

            int i                = 0;
            int vectorizedLength = length - (length % (dimensions * vectorSize));

            // Main vectorized loop
            for (; i < vectorizedLength; i += dimensions * vectorSize) {
                for (int d = 0; d < dimensions; d++) {
                    var span = new ReadOnlySpan<float>(ptr + i + d * vectorSize, vectorSize);
                    var vec  = new Vector<float>(span);
                    minVecs[d] = Vector.Min(minVecs[d], vec);
                    maxVecs[d] = Vector.Max(maxVecs[d], vec);
                }
            }

            // Reduce vectors to scalar values
            for (int d = 0; d < dimensions; d++) {
                min[d] = float.MaxValue;
                max[d] = float.MinValue;

                for (int j = 0; j < vectorSize; j++) {
                    min[d] = Math.Min(min[d], minVecs[d][j]);
                    max[d] = Math.Max(max[d], maxVecs[d][j]);
                }
            }

            ProcessRemainingElements(ptr, i, length, dimensions, min, max);
        }

        private static unsafe void ProcessScalar(float* ptr, int length, int dimensions, float[] min, float[] max) {
            for (int i = 0; i < length; i += dimensions) {
                for (int d = 0; d < dimensions; d++) {
                    min[d] = Math.Min(min[d], ptr[i + d]);
                    max[d] = Math.Max(max[d], ptr[i + d]);
                }
            }
        }

        private static unsafe void ProcessRemainingElements(float* ptr, int start, int length, int dimensions, float[] min, float[] max) {
            for (int i = start; i < length; i += dimensions) {
                for (int d = 0; d < dimensions; d++) {
                    min[d] = Math.Min(min[d], ptr[i + d]);
                    max[d] = Math.Max(max[d], ptr[i + d]);
                }
            }
        }
    }
}