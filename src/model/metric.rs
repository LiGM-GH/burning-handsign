use burn::{
    prelude::Backend,
    tensor::{Int, Tensor, Transaction},
    train::metric::{
        AccuracyInput, Adaptor, ItemLazy, LossInput, Metric, MetricEntry,
        MetricMetadata, Numeric,
        state::{FormatOptions, NumericMetricState},
    },
};
use derive_new::new;

const GUESS_BORDER: f32 = 0.5;

#[derive(new)]
pub struct BinaryClassificationOutput<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,

    /// The output.
    pub output: Tensor<B, 1>,

    /// The targets.
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> ItemLazy for BinaryClassificationOutput<B> {
    type ItemSync = BinaryClassificationOutput<B>;

    fn sync(self) -> Self::ItemSync {
        let [output, loss, targets] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        BinaryClassificationOutput {
            output: Tensor::from_data(output, device),
            loss: Tensor::from_data(loss, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}

#[derive(Default)]
pub struct TimesGuessedMetric<B: Backend> {
    state: NumericMetricState,
    _b: B,
}

impl<B: Backend> TimesGuessedMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(new)]
pub struct TimesGuessedInput<B: Backend> {
    tensor: Tensor<B, 1>,
    targets: Tensor<B, 1, Int>,
}

pub fn tensor_to_guesses<B: Backend>(
    tensor: Tensor<B, 1>,
) -> Tensor<B, 1, Int> {
    tensor.greater_elem(GUESS_BORDER).bool_not().int()
}

impl<B: Backend> Metric for TimesGuessedMetric<B> {
    type Input = TimesGuessedInput<B>;

    const NAME: &'static str = "TimesGuessed";

    fn update(
        &mut self,
        times: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> MetricEntry {
        let [batch_size] = times.tensor.dims();

        let real_times = tensor_to_guesses(times.tensor.clone())
            .sub(times.targets.clone())
            .powi_scalar(2)
            .sum();

        log::info!(
            "TimesGuessed: {}: {} whilst real data is {}",
            real_times.clone().into_scalar(),
            times.tensor,
            times.targets
        );

        let times = real_times
            .float()
            .div_scalar(times.targets.dims()[0] as f32)
            .into_data()
            .iter::<f64>()
            .next()
            .expect("Update failed on metric.rs/TimesGuessedMetric/update");

        self.state.update(
            times,
            batch_size,
            FormatOptions::new(Self::NAME).precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> Numeric for TimesGuessedMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

impl<B: Backend> Adaptor<TimesGuessedInput<B>>
    for BinaryClassificationOutput<B>
{
    fn adapt(&self) -> TimesGuessedInput<B> {
        TimesGuessedInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for BinaryClassificationOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
