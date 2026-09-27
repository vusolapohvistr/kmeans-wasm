# Benchmark images

Photographs used by `npm run bench:images`, which measures the packed entry
points on real image data rather than on random pixels.

Real images matter here because of how the reduction to distinct colours behaves.
Uniform random pixels are about 99% distinct, so they are the worst possible
input for it. Photographs are not: the standard test images in the colour
quantization literature are 7% to 58% distinct, and the four below are 6% to 46%.

| File | Size | Distinct | Source |
| --- | ---: | ---: | --- |
| `blue-marble.jpg` | 960 × 961 | 6.2% | Already vendored for the playground, from `docs/assets`. NASA, public domain. See `docs/assets/ATTRIBUTION.md`. |
| `harbour.jpg` | 640 × 480 | 36.6% | [Lorem Picsum](https://picsum.photos/), photo id 1015 |
| `coast.jpg` | 640 × 480 | 30.5% | [Lorem Picsum](https://picsum.photos/), photo id 1016 |
| `city.jpg` | 640 × 480 | 45.8% | [Lorem Picsum](https://picsum.photos/), photo id 164 |

The Lorem Picsum images are placeholders served from Unsplash. They are used here
as benchmark input only, are not redistributed anywhere users see, and can be
replaced with any JPEG by dropping a file in this directory; the harness decodes
every `.jpg` it finds.

To refresh them:

```sh
cd js_bench/images
for spec in "harbour:1015" "coast:1016" "city:164"; do
  name="${spec%%:*}"; id="${spec#*:}"
  curl -sSL -o "$name.jpg" "https://picsum.photos/id/$id/640/480"
done
```

`jpeg-js` decodes them. It is a pure-JavaScript JPEG decoder, so the harness needs
no native module and no image tooling on the machine.
