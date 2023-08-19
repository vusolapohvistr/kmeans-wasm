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

### K-means for any vector-space

To find the k-means centroids for any vector-space:

```javascript
import { kmeans } from 'kmeans-wasm';

// Sample data - an array of arrays where each inner array represents a point in the vector-space
const data = [
[1, 2],
[2, 3],
[3, 4],
[4, 5],
];

const k = 3; // Number of clusters
const max_iter = 1000; // Maximum number of iterations
const convergence_threshold = 0.001; // Convergence threshold

const result = kmeans(data, k, max_iter, convergence_threshold);

console.log(result);
```

### K-means for RGB color quantization

To find the k-means centroids of an RGB u8 slice for color quantization:

```javascript
import { kmeans_rgb } from 'kmeans-wasm';

// Sample data - Uint8Array of RGB components, where each component is a u8 value
const rgb_slice = new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255]);

const k = 3; // Number of clusters
const max_iter = 1000; // Maximum number of iterations
const convergence_threshold = 0.001; // Convergence threshold

const quantized_colors = kmeans_rgb(rgb_slice, k, max_iter, convergence_threshold);

console.log(quantized_colors);
```

## Comparison with skmeans

kmeans-wasm is faster.

You can test it yourself on <https://ycatbink0t.github.io/kmeans-web-comparison/>

## Contributing

Pull requests and issues are welcome. Please make sure to add tests for any new features or bug fixes.

## License

MIT License
