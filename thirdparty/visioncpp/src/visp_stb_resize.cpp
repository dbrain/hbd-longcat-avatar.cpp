// stb_image_resize implementation for the vendored visp image code.
//
// Deliberately ONLY the resize implementation: stb_image / stb_image_write are already
// instantiated by whichever TU includes image_io.hpp (which does
// `#define STB_IMAGE_IMPLEMENTATION`). Building this file into a static archive means the
// linker pulls it in only when stbir_* is otherwise unresolved, so linking the matte into a
// binary that already provides its own stb resize is a no-op rather than a duplicate symbol.
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>
