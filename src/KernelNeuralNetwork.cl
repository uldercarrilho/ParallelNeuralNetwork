__kernel void Sigmoide(__global float *ANeuronsOUT, float ASum)
{
    int i = get_global_id(0);
    ANeuronsOUT[i] = 1 / (1 - exp(-ASum));
}

__kernel void MultplyWeights(
    __global float *ANeurons, 
    __global float *AWeights,
    __global float *AResults)
{
    int i = get_global_id(0);
    AResults[i] = ANeurons[i] * AWeights[i];
}