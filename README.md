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

| Algorithm | k | 3 Dimensions | 10 Dimensions | 50 Dimensions |
|------------|---|--------------|---------------|---------------|
| skmeans | 2 | 16.73 | 34.28 | 249.82 |
| skmeans |10 | 75.29 | 348.19 | 1188.29 |
| skmeans |50 | 478.65 | 1185.82 | 3775.68 |
| kmeans-wasm| 2 | 12.96 | 21.87 | 119.22 |
| kmeans-wasm|10 | 27.37 | 81.18 | 422.66 |
| kmeans-wasm|50 | 87.59 | 310.45 | 1824.67 |

Caption: Comparison of average execution time (in milliseconds) for skmeans and kmeans-wasm with different dimensions and cluster sizes (k). The data size is 10,000 points, and the maximum number of iterations is 100.

### Comparison for RGB Data

| Algorithm    | k | 3 Dimensions |
|--------------|---|--------------|
| skmeans RGB   | 2 | 450.54       |
| skmeans RGB   | 10 | 8436.12     |
| skmeans RGB   | 50 | 51703.41    |
| kmeans_rgb RGB| 2 | 135.91       |
| kmeans_rgb RGB| 10 | 248.79      |
| kmeans_rgb RGB| 50 | 880.50      |

You can test it yourself on <https://ycatbink0t.github.io/kmeans-web-comparison/>

## Contributing

Pull requests and issues are welcome. Please make sure to add tests for any new features or bug fixes.

## License

MIT License
