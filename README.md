# MapManagerCore

MapManagerCore is a Python library that provides the core functionality for MapManager.

An example notebook is located under `/examples/`.

## Notes

This module is designed to be used by the web, and it contains async methods to allow for concurrent remote requests. However, since most scripts use local resources such as images and annotations, async functions are automatically converted to synchronous functions. This means that `await` is not required when using async functions outside of MapManagerCore.

For examples on how to interface with MapManagerCore, please refer to the example notebook.
