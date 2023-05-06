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

### Comparison for Data 1

| Algorithm | k | 3 Dimensions | 10 Dimensions | 50 Dimensions |
|-----------|---|--------------|---------------|---------------|
| skmeans    | 2 | 5.33         | 1.82          | 7.47          |
| skmeans    |10 | 6.54         | 12.34         | 25.97         |
| skmeans    |50 | 15.63        | 25.75         | 77.47         |
| kmeans-wasm| 2 | 1.92         | 1.30          | 5.49          |
| kmeans-wasm|10 | 0.94         | 2.23          | 8.82          |
| kmeans-wasm|50 | 1.98         | 4.46          | 16.46         |

### Comparison for Data 2

| Algorithm | k | 3 Dimensions | 10 Dimensions | 50 Dimensions |
|-----------|---|--------------|---------------|---------------|
| skmeans    | 2 | 4.87         | 2.37          | 7.22          |
| skmeans    |10 | 6.69         | 14.67         | 34.22         |
| skmeans    |50 | 12.72        | 25.79         | 84.56         |
| kmeans-wasm| 2 | 0.97         | 1.36          | 5.76          |
| kmeans-wasm|10 | 0.90         | 1.92          | 8.50          |
| kmeans-wasm|50 | 1.81         | 4.12          | 16.10         |

### Comparison for RGB Data

| Algorithm    | k | 3 Dimensions |
|--------------|---|--------------|
| skmeans RGB  | 2 | 450.54       |
| skmeans RGB  |10 | 8436.12      |
| skmeans RGB  |50 | 51703.41     |
| kmeans-wasm RGB| 2 | 135.91      |
| kmeans-wasm RGB|10 | 248.79      |
| kmeans-wasm RGB|50 | 880.50      |

### Comparison for RGB Data 2

| Algorithm    | k | 3 Dimensions |
|--------------|---|--------------|
| skmeans RGB  | 2 | 420.94       |
| skmeans RGB  |10 | 8149.42      |
| skmeans RGB  |50 | 51334.09     |
| kmeans-wasm RGB| 2 | 137.31      |
| kmeans-wasm RGB|10 | 266.48      |
| kmeans-wasm RGB|50 | 949.77      |

You can test it yourself on <https://ycatbink0t.github.io/kmeans-web-comparison/>

## Contributing

Pull requests and issues are welcome. Please make sure to add tests for any new features or bug fixes.

## License

MIT License
