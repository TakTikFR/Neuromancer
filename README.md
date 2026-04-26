# 🧠 Neuromancer

> A neural network framework built from scratch in Rust for learning, understanding, and eventually becoming something real.

---

## 🎯 Purpose

This project is **purely educational**. The goal is to deeply understand the mathematical and algorithmic foundations of machine learning by reimplementing them from the ground up, no magic, no black boxes.

The primary objective is to build a **Multi-Layer Perceptron (MLP)** capable of recognizing handwritten digits on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), while simultaneously learning and improving Rust skills through real, non-trivial code.

Beyond that first milestone, the ambition is to grow this project into something more substantial and potentially evolve it into a **proper, usable Rust ML library**.

---

## 🦀 Why Rust

Rust is a relatively recent language, and one that will likely become increasingly useful over the coming years. It allows a decent level of abstraction while remaining extremely performant which makes it a compelling choice for this kind of project.

Beyond the technical side, this project is also simply a way to get better at Rust by working on something concrete and challenging.

---

## ⚙️ Technical Stack

The project uses [`candle-core`](https://github.com/huggingface/candle) (by Hugging Face) as the **sole external ML dependency**, providing the `Tensor` struct, math operations, device abstraction, and error handling.

Everything else layers, activations, loss functions, optimizers and the training loop is implemented **from scratch**.

---

## 🗂️ Project Structure

```
Neuromancer/
├── Cargo.lock
├── Cargo.toml
├── README.md
└── src
    ├── activations
    │   ├── relu.rs
    │   └── softmax.rs
    ├── activations.rs
    ├── data
    │   ├── dataloader.rs
    │   └── mnist.rs
    ├── data.rs
    ├── layers
    │   ├── linear.rs
    │   └── sequential.rs
    ├── layers.rs
    ├── lib.rs
    ├── loss
    │   └── cross_entropy.rs
    ├── loss.rs
    ├── main.rs
    ├── optimizers
    │   ├── adam.rs
    │   └── sgd.rs
    ├── optimizers.rs
    └── tensor.rs
```

---

## ✅ Roadmap

### 🔧 General

- [x] Implement an MLP for the MNIST dataset
- [ ] Implement a UNet architecture

### 🧱 Layers

- [x] Linear
- [ ] Conv2D
- [ ] Dropout
- [ ] BatchNorm

### ⚡ Activations

- [x] ReLU
- [x] Softmax
- [ ] Sigmoid
- [ ] Tanh
- [ ] GELU

### 📉 Loss Functions

- [x] CrossEntropy
- [ ] MSE
- [ ] L1

### 🏃 Optimizers

- [x] SGD
- [x] Adam
- [ ] AdamW
- [ ] Adagrad
