__kernel void multiply(
    __global float* inputs,
    __global float* weights,
    __global float* results,
    uint inputOffset,
    uint inputSize,
    uint outputSize)
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
    __global float* multiplies,
    __global float* outputs,
    uint inputSize,
    uint outputSize)
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
    __global float* outputs,
    __global float* samples,
    __global float* results,
    uint outputSize,
    uint samplesOffset)
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
    __global float* hidden,
    __global float* deltaOutput,
    __global float* weights,
    __global float* results,
    uint hiddenSize,
    uint outputSize)
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
    __global float* neurons,
    __global float* delta,
    uint neuronOffset,
    uint neuronSize,
    uint deltaSize,
    float eta)
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