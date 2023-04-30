# kmeans-wasm

A WebAssembly implementation of the k-means clustering algorithm for color quantization and general vector-space clustering.

## Features

Fast k-means clustering using the Hamerly algorithm   
Can be used for color quantization in image processing  
Works with any vector-space  
Exports both JavaScript and TypeScript bindings  
## Installation

 npm install kmeans-wasm 

## Usage

### Color Quantization

```javascript
import { kmeans_rgb } from 'kmeans-wasm';

const k = 16; // Number of clusters (colors)
const max_iter = 1000; // Maximum number of iterations
const convergence_threshold = 0; // The threshold to determine when the centroids have converged
const rgb_slice = [...]; // Uint8Array of RGB components

const centroids = kmeans_rgb(k, max_iter, convergence_threshold, rgb_slice);

// Use the centroids (quantized colors) for further processing
```

### General Vector-space Clustering

```javascript
import { kmeans } from 'kmeans-wasm';

const k = 5; // Number of clusters
const max_iter = 1000; // Maximum number of iterations
const convergence_threshold = 0; // The threshold to determine when the centroids have converged
const data = [
[1.0, 2.0],
[1.1, 2.1],
// ...
];

const centroids = kmeans(k, max_iter, convergence_threshold, data);

// Use the centroids for further processing
```

## Contributing

Pull requests and issues are welcome. Please make sure to add tests for any new features or bug fixes.

## License

MIT License