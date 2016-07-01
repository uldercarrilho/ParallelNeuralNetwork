// First naive implementation
__kernel void myGEMM1(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {
    
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
 
    // Compute a single element (loop over K)
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += A[globalRow * K + i] * B[i * N + globalCol];
    }
 
    // Store the result
    C[globalRow*N + globalCol] = sum;
    //C[globalRow*N + globalCol] = 1 / (1 + exp(-sum));
}

__kernel void active(const int M, const int N, const int K, const uint offsetA,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {
    
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
 
    // Compute a single element (loop over K)
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += A[offsetA + globalRow * K + i] * B[i * N + globalCol];
    }
 
    // Store the result
    C[globalRow*N + globalCol] = sum;
    //C[globalRow*N + globalCol] = 1 / (1 + exp(-sum));
}

__kernel void multiply(
    __global const float *inputs,
    __global const float *weights,
    __global float *results,
    uint const inputOffset,
    uint const inputSize,
    uint const outputSize)
{
    uint inputId = get_global_id(0);
    uint outputId = get_global_id(1);
    
    if (inputId >= inputSize)
        return;
    
    if (outputId >= outputSize)
        return;

    uint id = inputId * outputSize + outputId;

    results[id] = inputs[inputOffset + inputId] * weights[id];
}

__kernel void sigmoide(
    __global const float *multiplies,
    __global float *outputs,
    uint const inputSize,
    uint const outputSize)
{
    uint outputId = get_global_id(0);
    if (outputId >= outputSize)
        return;

    float sum = 0.0f;
    uint id;
    for (uint i = 0; i < inputSize; i++)
    {
        id = i * outputSize + outputId;
        sum += multiplies[id];
    }

    outputs[outputId] = 1 / (1 + exp(-sum));
}

__kernel void calculateDeltaOutput(
    __global const float *outputs,
    __global const float *samples,
    __global float *results,
    uint const outputSize,
    uint const samplesOffset)
{
    uint outputId = get_global_id(0);
    if (outputId >= outputSize)
        return;

    results[outputId] = 
        outputs[outputId] * 
        (1 - outputs[outputId]) * 
        (samples[samplesOffset + outputId] - outputs[outputId]);
}

__kernel void calculateDeltaHidden(
    __global const float *hidden,
    __global const float *deltaOutput,
    __global const float *weights,
    __global float *results,
    uint const hiddenSize,
    uint const outputSize)
{
    uint hiddenId = get_global_id(0);
    if (hiddenId >= hiddenSize)
        return;

    float sum = 0.0f;
    uint id;
    for (uint i = 0; i < outputSize; i++)
    {
        id = hiddenId * outputSize + i;
        sum += deltaOutput[i] * weights[id];
    }

    results[hiddenId] = hidden[hiddenId] * (1 - hidden[hiddenId]) * sum;
}

__kernel void updateWeights(
    __global float* weights,
    __global const float *neurons,
    __global const float *delta,
    uint const neuronOffset,
    uint const neuronSize,
    uint const deltaSize,
    float const eta)
{
    uint neuronId = get_global_id(0);
    uint deltaId  = get_global_id(1);

    if (neuronId >= neuronSize)
        return;

    if (deltaId >= deltaSize)
        return;

    uint id = neuronId * deltaSize + deltaId;
    weights[id] += eta * delta[deltaId] * neurons[neuronOffset + neuronId];
}

__kernel void params(
    __global float* result,
    __global float* param1,
    __global float* param2,
    __global float* param3,
    __global float* param4,
    __global float* param5,
    __global float* param6,
    __global float* param7,
    __global float* param8,
    __global float* param9)
{
    uint id = get_global_id(0);

    result[id] = 
        param1[id] + 
        param2[id] + 
        param3[id] + 
        param4[id] + 
        param5[id] + 
        param6[id] + 
        param7[id] + 
        param8[id] + 
        param9[id];
}