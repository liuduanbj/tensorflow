export class MnistData {
    load(): Promise<void>;
    nextTrainBatch(batchSize: number): {
        xs: Tensor2D;
        labels: Tensor2D;
    };
    nextTestBatch(batchSize: number): {
        xs: Tensor2D;
        labels: Tensor2D;
    };
} 