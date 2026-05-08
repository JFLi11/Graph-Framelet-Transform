This repository provides an implementation of the **Fast Graph Framelet Transform** based on hierarchical graph partitioning. The algorithm is designed to analyze graph-structured data by decomposing signals into multi-scale components while maintaining the **Tight Frame** property for perfect reconstruction.

## 📜 Source and Attribution

The core algorithmic logic, including the transform operators in `binary_haar_graph_framelets.py`, is based on the methodology described in the following research:

**Reference Paper: [[Permutation equivariant graph framelets for heterophilous graph learning](https://ieeexplore.ieee.org/document/10466590)]**

This implementation specifically focuses on:
1. Framelet Construction: Providing a streamlined framework for constructing Haar-type and other tight graph framelets based on hierarchical graph partitioning, ensuring theoretical rigor and perfect reconstruction.
2. High-Performance Computation: A user-friendly, modular implementation optimized for large-scale graphs, featuring seamless GPU acceleration via PyTorch for speedups in decomposition and reconstruction.
---

## 🚀 Project Structure

* **`binary_haar_graph_framelets.py`**: The main implementation of the `GraphFrameletTransform` class. Includes methods for Fast Forward Transform (FFT), Inverse Fast Transform (IFFT), and explicit framelet matrix calculation.
* **`main_demo.py`**: A complete demonstration script that generates a random graph, computes the transform, and verifies the Tight Frame properties.

---

## 🛠 Installation

Requires Python 3.8+ and the following packages:

```bash
pip install torch numpy scipy networkx scikit-network

---

## References

If you find this code useful in your research, please consider citing the following paper:

```bibtex
@article{li2024permutation,
  title={Permutation equivariant graph framelets for heterophilous graph learning},
  author={Li, Jianfei and Zheng, Ruigang and Feng, Han and Li, Ming and Zhuang, Xiaosheng},
  journal={IEEE Transactions on neural networks and learning systems},
  volume={35},
  number={9},
  pages={11634--11648},
  year={2024},
  publisher={IEEE}
}

---
