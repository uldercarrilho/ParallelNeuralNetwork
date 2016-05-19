__kernel void multiply(
    __global float* inputs,
    __global float* weights,
    __global float* results,
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
    results[id] = inputs[inputId] * weights[id];
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
    uint outputSize)
{
    uint outputId = get_global_id(0);
    if (outputId >= outputSize)
        return;

    results[outputId] = outputs[outputId] * (1 - outputs[outputId]) * (samples[outputId] - outputs[outputId]);
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
    weights[id] += eta * delta[deltaId] * neurons[neuronId];
}