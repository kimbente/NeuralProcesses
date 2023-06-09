# NeuralProcesses

Implementation of Neural Processes in Pytorch and synthetic examples including:
- 1D regression of sin functions
- 1D regression of discontinous sin functions
- 1D regression of Gaussian Process functions

## Citations

**Conditional Neural Processes**: Garnelo M, Rosenbaum D, Maddison CJ, Ramalho T, Saxton D, Shanahan M, Teh YW, Rezende DJ, Eslami SM. Conditional Neural Processes. In International Conference on Machine Learning 2018.

**Neural Processes**: Garnelo, M., Schwarz, J., Rosenbaum, D., Viola, F., Rezende, D.J., Eslami, S.M. and Teh, Y.W. Neural processes. ICML Workshop on Theoretical Foundations and Applications of Deep Generative Models 2018.

**Attentive Neural Processes**: Kim, H., Mnih, A., Schwarz, J., Garnelo, M., Eslami, A., Rosenbaum, D., Vinyals, O. and Teh, Y.W. Attentive Neural Processes. In International Conference on Learning Representations 2019.

## Resources

- [Official Neural Process implementations by Deepmind](https://github.com/deepmind/neural-processes)
- [Emilien Dupont Neural Processes implementation](https://github.com/EmilienDupont/neural-processes)
- [Kaspar Märtens blogpost and R implementation](https://kasparmartens.rbind.io/post/np/)
  - Provides intuition for priors in NPs.
- [Wessel Brunsima neuralprocesses package](https://github.com/wesselb/neuralprocesses) and [documentation]()
- [Yann Dubois et al. blog/website](https://yanndubs.github.io/Neural-Process-Family/text/Intro.html)
- [Xuesong Wangs' implementation](https://github.com/xuesongwang/Neural-Process-Family)
- [Infinite Norm blogpost contrasting NPs and VAEs and PyTorch implementation](https://chrisorm.github.io/NGP.html)
  - Implementation not consistent with papers.
  - VAEs learn the a latent representation for data points. NPs learn a latent representation for functions via data sets.

## Contact

kim.bente@sydney.edu.au
