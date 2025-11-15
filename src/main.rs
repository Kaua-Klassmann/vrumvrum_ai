use std::fs;

use tch::{
    Device, Kind, Reduction, Tensor,
    nn::{self, OptimizerConfig},
};

fn read_dataset() -> (Vec<f32>, Vec<i32>) {
    let mut features = Vec::new();
    let mut targets = Vec::new();

    let folders = vec!["aviao", "carro", "moto", "navio"];

    for (label, folder) in folders.iter().enumerate() {
        let files = fs::read_dir(format!("dataset/{}", folder))
            .unwrap()
            .map(|file| file.unwrap());

        for file in files {
            let img = image::open(file.path())
                .unwrap()
                .to_rgb8()
                .into_raw()
                .into_iter()
                .map(|byte| byte as f32);

            features.extend(img);
            targets.push(label as i32);
        }
    }

    (features, targets)
}

fn main() {
    // setup
    tch::manual_seed(42);
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let root = &vs.root();

    // read dataset
    let (features, targets) = read_dataset();

    // setup tensors
    let mut features = Tensor::from_slice(&features)
        .reshape([features.len() as i64 / (640 * 640 * 3), 3, 640, 640])
        .to_kind(Kind::Float)
        .to_device(device);

    features /= 255.;

    let targets = Tensor::from_slice(&targets)
        .reshape([targets.len() as i64])
        .to_kind(Kind::Int64)
        .to_device(device);

    let perm = Tensor::randperm(targets.size()[0], (Kind::Int64, device));

    let features = features.index_select(0, &perm);
    let targets = targets.index_select(0, &perm);

    let train_range = (targets.size()[0] as f32 * 0.8) as i64;

    let x_train = features.narrow(0, 0, train_range);
    let y_train = targets.narrow(0, 0, train_range);

    let x_test = features.narrow(0, train_range, features.size()[0] - train_range);
    let y_test = targets.narrow(0, train_range, targets.size()[0] - train_range);

    drop(features);
    drop(targets);

    // create model
    let model = nn::seq_t()
        // conv 1
        .add(nn::conv2d(root, 3, 4, 3, Default::default()))
        .add(nn::batch_norm2d(root, 4, Default::default()))
        .add_fn(|xs| xs.leaky_relu())
        .add_fn(|xs| xs.max_pool2d(3, 2, 0, 1, false))
        // conv 2
        .add(nn::conv2d(root, 4, 8, 3, Default::default()))
        .add(nn::batch_norm2d(root, 8, Default::default()))
        .add_fn(|xs| xs.leaky_relu())
        .add_fn(|xs| xs.max_pool2d(3, 2, 0, 1, false))
        // conv 3
        .add(nn::conv2d(root, 8, 16, 3, Default::default()))
        .add(nn::batch_norm2d(root, 16, Default::default()))
        .add_fn(|xs| xs.leaky_relu())
        .add_fn(|xs| xs.max_pool2d(3, 2, 0, 1, false))
        // conv 4
        .add(nn::conv2d(root, 16, 32, 3, Default::default()))
        .add(nn::batch_norm2d(root, 32, Default::default()))
        .add_fn(|xs| xs.leaky_relu())
        .add_fn(|xs| xs.max_pool2d(3, 4, 0, 1, false))
        // flatten
        .add_fn(|xs| xs.flatten(1, -1))
        // linear 1
        .add(nn::linear(root, 32 * 19 * 19, 64, Default::default()))
        .add(nn::batch_norm1d(root, 64, Default::default()))
        .add_fn(|xs| xs.leaky_relu())
        .add_fn_t(|xs, train| xs.dropout(0.25, train))
        // linear 2
        .add(nn::linear(root, 64, 32, Default::default()))
        .add(nn::batch_norm1d(root, 32, Default::default()))
        .add_fn(|xs| xs.leaky_relu())
        .add_fn_t(|xs, train| xs.dropout(0.25, train))
        // out
        .add(nn::linear(root, 32, 4, Default::default()));

    let mut opt = nn::AdamW {
        wd: 1e-4,
        ..Default::default()
    }
    .build(&vs, 1e-3)
    .unwrap();

    // train model
    for epoch in 0..=80 {
        let y_pred = x_train.apply_t(&model, true);

        let loss = y_pred.cross_entropy_loss::<Tensor>(&y_train, None, Reduction::Mean, -100, 0.);

        opt.backward_step(&loss);

        if epoch % 1 == 0 {
            let y_pred = x_test.apply_t(&model, false).argmax(1, false);

            let acc = y_pred.accuracy_for_logits(&y_test);

            println!("Epoch {} - Acurácia {:?}", epoch, acc);
        }
    }

    vs.save("model.ot").unwrap();
}
