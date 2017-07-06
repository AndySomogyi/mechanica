# Mechanica

Mechanica is a mechanistic model declaration language and simulation engine for physically motivated phenomena that couple mechanical and chemical processes

Mechanica is designed to represent:

* Deformable surfaces
* Chemical fields
* Particles and composite particles with local state
* Transformation processes such as chemical reactions
* Collision detection between hard and soft objects
* Complex fluids
* Continuous control systems

# Build Instructions

Mechanica uses git submodules for dependencies, you need recursively clone the repository initally: 

```
git clone --recursive https://github.com/AndySomogyi/mechanica.git
```

Mechanica relies on a few external depencencies that are very standard and easy enough to get. You can grab these from brew:

```
brew install glfw
brew install assimp
brew install llvm
brew install fftw
brew install libpng
```
