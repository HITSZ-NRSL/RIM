#include "marching_cube.h"

#include "params/params.h"
#include "utils/utils.h"
#define TINYPLY_IMPLEMENTATION
#include "tinyply/tinyply.h"

using namespace std;

torch::Tensor TRIANGLE_TABLE = torch::tensor(
    {{12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {0, 8, 3, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {0, 1, 9, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {1, 8, 3, 9, 8, 1, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {1, 2, 10, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {0, 8, 3, 1, 2, 10, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {9, 2, 10, 0, 2, 9, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {2, 8, 3, 2, 10, 8, 10, 9, 8, 12, 12, 12, 12, 12, 12},
     {3, 11, 2, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {0, 11, 2, 8, 11, 0, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {1, 9, 0, 2, 3, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {1, 11, 2, 1, 9, 11, 9, 8, 11, 12, 12, 12, 12, 12, 12},
     {3, 10, 1, 11, 10, 3, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {0, 10, 1, 0, 8, 10, 8, 11, 10, 12, 12, 12, 12, 12, 12},
     {3, 9, 0, 3, 11, 9, 11, 10, 9, 12, 12, 12, 12, 12, 12},
     {9, 8, 10, 10, 8, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {4, 7, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {4, 3, 0, 7, 3, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {0, 1, 9, 8, 4, 7, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {4, 1, 9, 4, 7, 1, 7, 3, 1, 12, 12, 12, 12, 12, 12},
     {1, 2, 10, 8, 4, 7, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {3, 4, 7, 3, 0, 4, 1, 2, 10, 12, 12, 12, 12, 12, 12},
     {9, 2, 10, 9, 0, 2, 8, 4, 7, 12, 12, 12, 12, 12, 12},
     {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, 12, 12, 12},
     {8, 4, 7, 3, 11, 2, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {11, 4, 7, 11, 2, 4, 2, 0, 4, 12, 12, 12, 12, 12, 12},
     {9, 0, 1, 8, 4, 7, 2, 3, 11, 12, 12, 12, 12, 12, 12},
     {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, 12, 12, 12},
     {3, 10, 1, 3, 11, 10, 7, 8, 4, 12, 12, 12, 12, 12, 12},
     {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, 12, 12, 12},
     {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, 12, 12, 12},
     {4, 7, 11, 4, 11, 9, 9, 11, 10, 12, 12, 12, 12, 12, 12},
     {9, 5, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {9, 5, 4, 0, 8, 3, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {0, 5, 4, 1, 5, 0, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {8, 5, 4, 8, 3, 5, 3, 1, 5, 12, 12, 12, 12, 12, 12},
     {1, 2, 10, 9, 5, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {3, 0, 8, 1, 2, 10, 4, 9, 5, 12, 12, 12, 12, 12, 12},
     {5, 2, 10, 5, 4, 2, 4, 0, 2, 12, 12, 12, 12, 12, 12},
     {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, 12, 12, 12},
     {9, 5, 4, 2, 3, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {0, 11, 2, 0, 8, 11, 4, 9, 5, 12, 12, 12, 12, 12, 12},
     {0, 5, 4, 0, 1, 5, 2, 3, 11, 12, 12, 12, 12, 12, 12},
     {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, 12, 12, 12},
     {10, 3, 11, 10, 1, 3, 9, 5, 4, 12, 12, 12, 12, 12, 12},
     {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, 12, 12, 12},
     {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, 12, 12, 12},
     {5, 4, 8, 5, 8, 10, 10, 8, 11, 12, 12, 12, 12, 12, 12},
     {9, 7, 8, 5, 7, 9, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {9, 3, 0, 9, 5, 3, 5, 7, 3, 12, 12, 12, 12, 12, 12},
     {0, 7, 8, 0, 1, 7, 1, 5, 7, 12, 12, 12, 12, 12, 12},
     {1, 5, 3, 3, 5, 7, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {9, 7, 8, 9, 5, 7, 10, 1, 2, 12, 12, 12, 12, 12, 12},
     {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, 12, 12, 12},
     {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, 12, 12, 12},
     {2, 10, 5, 2, 5, 3, 3, 5, 7, 12, 12, 12, 12, 12, 12},
     {7, 9, 5, 7, 8, 9, 3, 11, 2, 12, 12, 12, 12, 12, 12},
     {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, 12, 12, 12},
     {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, 12, 12, 12},
     {11, 2, 1, 11, 1, 7, 7, 1, 5, 12, 12, 12, 12, 12, 12},
     {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, 12, 12, 12},
     {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0},
     {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0},
     {11, 10, 5, 7, 11, 5, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {10, 6, 5, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {0, 8, 3, 5, 10, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {9, 0, 1, 5, 10, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {1, 8, 3, 1, 9, 8, 5, 10, 6, 12, 12, 12, 12, 12, 12},
     {1, 6, 5, 2, 6, 1, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {1, 6, 5, 1, 2, 6, 3, 0, 8, 12, 12, 12, 12, 12, 12},
     {9, 6, 5, 9, 0, 6, 0, 2, 6, 12, 12, 12, 12, 12, 12},
     {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, 12, 12, 12},
     {2, 3, 11, 10, 6, 5, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {11, 0, 8, 11, 2, 0, 10, 6, 5, 12, 12, 12, 12, 12, 12},
     {0, 1, 9, 2, 3, 11, 5, 10, 6, 12, 12, 12, 12, 12, 12},
     {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, 12, 12, 12},
     {6, 3, 11, 6, 5, 3, 5, 1, 3, 12, 12, 12, 12, 12, 12},
     {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, 12, 12, 12},
     {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, 12, 12, 12},
     {6, 5, 9, 6, 9, 11, 11, 9, 8, 12, 12, 12, 12, 12, 12},
     {5, 10, 6, 4, 7, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {4, 3, 0, 4, 7, 3, 6, 5, 10, 12, 12, 12, 12, 12, 12},
     {1, 9, 0, 5, 10, 6, 8, 4, 7, 12, 12, 12, 12, 12, 12},
     {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, 12, 12, 12},
     {6, 1, 2, 6, 5, 1, 4, 7, 8, 12, 12, 12, 12, 12, 12},
     {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, 12, 12, 12},
     {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, 12, 12, 12},
     {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9},
     {3, 11, 2, 7, 8, 4, 10, 6, 5, 12, 12, 12, 12, 12, 12},
     {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, 12, 12, 12},
     {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, 12, 12, 12},
     {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6},
     {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, 12, 12, 12},
     {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11},
     {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7},
     {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, 12, 12, 12},
     {10, 4, 9, 6, 4, 10, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {4, 10, 6, 4, 9, 10, 0, 8, 3, 12, 12, 12, 12, 12, 12},
     {10, 0, 1, 10, 6, 0, 6, 4, 0, 12, 12, 12, 12, 12, 12},
     {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, 12, 12, 12},
     {1, 4, 9, 1, 2, 4, 2, 6, 4, 12, 12, 12, 12, 12, 12},
     {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, 12, 12, 12},
     {0, 2, 4, 4, 2, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {8, 3, 2, 8, 2, 4, 4, 2, 6, 12, 12, 12, 12, 12, 12},
     {10, 4, 9, 10, 6, 4, 11, 2, 3, 12, 12, 12, 12, 12, 12},
     {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, 12, 12, 12},
     {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, 12, 12, 12},
     {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1},
     {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, 12, 12, 12},
     {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1},
     {3, 11, 6, 3, 6, 0, 0, 6, 4, 12, 12, 12, 12, 12, 12},
     {6, 4, 8, 11, 6, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {7, 10, 6, 7, 8, 10, 8, 9, 10, 12, 12, 12, 12, 12, 12},
     {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, 12, 12, 12},
     {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, 12, 12, 12},
     {10, 6, 7, 10, 7, 1, 1, 7, 3, 12, 12, 12, 12, 12, 12},
     {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, 12, 12, 12},
     {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9},
     {7, 8, 0, 7, 0, 6, 6, 0, 2, 12, 12, 12, 12, 12, 12},
     {7, 3, 2, 6, 7, 2, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, 12, 12, 12},
     {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7},
     {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11},
     {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, 12, 12, 12},
     {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6},
     {0, 9, 1, 11, 6, 7, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, 12, 12, 12},
     {7, 11, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {7, 6, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {3, 0, 8, 11, 7, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {0, 1, 9, 11, 7, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {8, 1, 9, 8, 3, 1, 11, 7, 6, 12, 12, 12, 12, 12, 12},
     {10, 1, 2, 6, 11, 7, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {1, 2, 10, 3, 0, 8, 6, 11, 7, 12, 12, 12, 12, 12, 12},
     {2, 9, 0, 2, 10, 9, 6, 11, 7, 12, 12, 12, 12, 12, 12},
     {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, 12, 12, 12},
     {7, 2, 3, 6, 2, 7, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {7, 0, 8, 7, 6, 0, 6, 2, 0, 12, 12, 12, 12, 12, 12},
     {2, 7, 6, 2, 3, 7, 0, 1, 9, 12, 12, 12, 12, 12, 12},
     {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, 12, 12, 12},
     {10, 7, 6, 10, 1, 7, 1, 3, 7, 12, 12, 12, 12, 12, 12},
     {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, 12, 12, 12},
     {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, 12, 12, 12},
     {7, 6, 10, 7, 10, 8, 8, 10, 9, 12, 12, 12, 12, 12, 12},
     {6, 8, 4, 11, 8, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {3, 6, 11, 3, 0, 6, 0, 4, 6, 12, 12, 12, 12, 12, 12},
     {8, 6, 11, 8, 4, 6, 9, 0, 1, 12, 12, 12, 12, 12, 12},
     {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, 12, 12, 12},
     {6, 8, 4, 6, 11, 8, 2, 10, 1, 12, 12, 12, 12, 12, 12},
     {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, 12, 12, 12},
     {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, 12, 12, 12},
     {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3},
     {8, 2, 3, 8, 4, 2, 4, 6, 2, 12, 12, 12, 12, 12, 12},
     {0, 4, 2, 4, 6, 2, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, 12, 12, 12},
     {1, 9, 4, 1, 4, 2, 2, 4, 6, 12, 12, 12, 12, 12, 12},
     {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, 12, 12, 12},
     {10, 1, 0, 10, 0, 6, 6, 0, 4, 12, 12, 12, 12, 12, 12},
     {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3},
     {10, 9, 4, 6, 10, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {4, 9, 5, 7, 6, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {0, 8, 3, 4, 9, 5, 11, 7, 6, 12, 12, 12, 12, 12, 12},
     {5, 0, 1, 5, 4, 0, 7, 6, 11, 12, 12, 12, 12, 12, 12},
     {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, 12, 12, 12},
     {9, 5, 4, 10, 1, 2, 7, 6, 11, 12, 12, 12, 12, 12, 12},
     {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, 12, 12, 12},
     {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, 12, 12, 12},
     {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6},
     {7, 2, 3, 7, 6, 2, 5, 4, 9, 12, 12, 12, 12, 12, 12},
     {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, 12, 12, 12},
     {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, 12, 12, 12},
     {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8},
     {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, 12, 12, 12},
     {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4},
     {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10},
     {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, 12, 12, 12},
     {6, 9, 5, 6, 11, 9, 11, 8, 9, 12, 12, 12, 12, 12, 12},
     {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, 12, 12, 12},
     {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, 12, 12, 12},
     {6, 11, 3, 6, 3, 5, 5, 3, 1, 12, 12, 12, 12, 12, 12},
     {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, 12, 12, 12},
     {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10},
     {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5},
     {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, 12, 12, 12},
     {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, 12, 12, 12},
     {9, 5, 6, 9, 6, 0, 0, 6, 2, 12, 12, 12, 12, 12, 12},
     {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8},
     {1, 5, 6, 2, 1, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6},
     {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, 12, 12, 12},
     {0, 3, 8, 5, 6, 10, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {10, 5, 6, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {11, 5, 10, 7, 5, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {11, 5, 10, 11, 7, 5, 8, 3, 0, 12, 12, 12, 12, 12, 12},
     {5, 11, 7, 5, 10, 11, 1, 9, 0, 12, 12, 12, 12, 12, 12},
     {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, 12, 12, 12},
     {11, 1, 2, 11, 7, 1, 7, 5, 1, 12, 12, 12, 12, 12, 12},
     {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, 12, 12, 12},
     {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, 12, 12, 12},
     {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2},
     {2, 5, 10, 2, 3, 5, 3, 7, 5, 12, 12, 12, 12, 12, 12},
     {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, 12, 12, 12},
     {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, 12, 12, 12},
     {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2},
     {1, 3, 5, 3, 7, 5, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {0, 8, 7, 0, 7, 1, 1, 7, 5, 12, 12, 12, 12, 12, 12},
     {9, 0, 3, 9, 3, 5, 5, 3, 7, 12, 12, 12, 12, 12, 12},
     {9, 8, 7, 5, 9, 7, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {5, 8, 4, 5, 10, 8, 10, 11, 8, 12, 12, 12, 12, 12, 12},
     {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, 12, 12, 12},
     {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, 12, 12, 12},
     {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4},
     {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, 12, 12, 12},
     {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11},
     {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5},
     {9, 4, 5, 2, 11, 3, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, 12, 12, 12},
     {5, 10, 2, 5, 2, 4, 4, 2, 0, 12, 12, 12, 12, 12, 12},
     {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9},
     {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, 12, 12, 12},
     {8, 4, 5, 8, 5, 3, 3, 5, 1, 12, 12, 12, 12, 12, 12},
     {0, 4, 5, 1, 0, 5, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, 12, 12, 12},
     {9, 4, 5, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {4, 11, 7, 4, 9, 11, 9, 10, 11, 12, 12, 12, 12, 12, 12},
     {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, 12, 12, 12},
     {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, 12, 12, 12},
     {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4},
     {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, 12, 12, 12},
     {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3},
     {11, 7, 4, 11, 4, 2, 2, 4, 0, 12, 12, 12, 12, 12, 12},
     {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, 12, 12, 12},
     {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, 12, 12, 12},
     {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7},
     {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10},
     {1, 10, 2, 8, 7, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {4, 9, 1, 4, 1, 7, 7, 1, 3, 12, 12, 12, 12, 12, 12},
     {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, 12, 12, 12},
     {4, 0, 3, 7, 4, 3, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {4, 8, 7, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {9, 10, 8, 10, 11, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {3, 0, 9, 3, 9, 11, 11, 9, 10, 12, 12, 12, 12, 12, 12},
     {0, 1, 10, 0, 10, 8, 8, 10, 11, 12, 12, 12, 12, 12, 12},
     {3, 1, 10, 11, 3, 10, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {1, 2, 11, 1, 11, 9, 9, 11, 8, 12, 12, 12, 12, 12, 12},
     {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, 12, 12, 12},
     {0, 2, 11, 8, 0, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {3, 2, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {2, 3, 8, 2, 8, 10, 10, 8, 9, 12, 12, 12, 12, 12, 12},
     {9, 10, 2, 0, 9, 2, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, 12, 12, 12},
     {1, 10, 2, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {1, 3, 8, 9, 1, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {0, 9, 1, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {0, 3, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12},
     {12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12}});

// index here represented in binary: 100->4
torch::Tensor EDGE_INDEX_PAIRS = torch::tensor({{0, 1},
                                                {1, 2},
                                                {2, 3},
                                                {3, 0},
                                                {4, 5},
                                                {5, 6},
                                                {6, 7},
                                                {7, 4},
                                                {0, 4},
                                                {1, 5},
                                                {2, 6},
                                                {3, 7},
                                                {8, 8}});

torch::Tensor cal_vertex_config(const torch::Tensor &voxel_vertex_sdf,
                                float isovalue) {
  // xyz:1<<index
  return (voxel_vertex_sdf.select(1, 0) < isovalue) |
         (voxel_vertex_sdf.select(1, 1) < isovalue) * 2 |
         (voxel_vertex_sdf.select(1, 2) < isovalue) * 4 |
         (voxel_vertex_sdf.select(1, 3) < isovalue) * 8 |
         (voxel_vertex_sdf.select(1, 4) < isovalue) * 16 |
         (voxel_vertex_sdf.select(1, 5) < isovalue) * 32 |
         (voxel_vertex_sdf.select(1, 6) < isovalue) * 64 |
         (voxel_vertex_sdf.select(1, 7) < isovalue) * 128;
}

torch::Tensor marching_cube(torch::Tensor _grid_sdf, torch::Tensor _grid_xyz,
                            float _isovalue) {

  /// [voxel_num*n,1]
  auto mesh_voxel_cfg = cal_vertex_config(_grid_sdf, _isovalue);
  /// [voxel_num*n,15]
  auto voxel_edge_index = TRIANGLE_TABLE.index({mesh_voxel_cfg});

  // [voxel_num*n,5*3*2,1]
  auto voxel_face_edge_pair_index =
      EDGE_INDEX_PAIRS.index({voxel_edge_index.view({-1})}).view({-1, 30, 1});
  // [n*5*3*2,2]
  auto voxel_pair_index_ =
      torch::cat(
          {torch::arange(0, voxel_face_edge_pair_index.size(0), *p_device)
               .view({-1, 1, 1})
               .expand({-1, 30, 1}),
           voxel_face_edge_pair_index},
          -1)
          .view({-1, 2})
          .to(torch::kInt64);

  // [n,8+1]
  _grid_sdf =
      torch::cat({_grid_sdf, torch::zeros({_grid_sdf.size(0), 1}, *p_device)},
                 1); // coresponding coef should be nan
  // [n*5*3,2]
  auto voxel_face_edge_pair_sdf = _grid_sdf
                                      .index({voxel_pair_index_.select(1, 0),
                                              voxel_pair_index_.select(1, 1)})
                                      .view({-1, 2});
  // [n*5*3,1]
  auto voxel_face_edge_coef =
      ((_isovalue - voxel_face_edge_pair_sdf.select(1, 0)) /
       (voxel_face_edge_pair_sdf.select(1, 1) -
        voxel_face_edge_pair_sdf.select(1, 0)))
          .view({-1, 1});
  // [n,8,3]
  _grid_xyz = _grid_xyz.view({-1, 8, 3});
  // [n,8+1,3]
  _grid_xyz = torch::cat(
      {_grid_xyz, torch::zeros({_grid_xyz.size(0), 1, 3}, *p_device)}, 1);
  // [n*5*3,2,3]
  auto voxel_face_edge_pair_xyz = _grid_xyz
                                      .index({voxel_pair_index_.select(1, 0),
                                              voxel_pair_index_.select(1, 1)})
                                      .view({-1, 2, 3});
  // [n*5,3,3]
  auto voxel_face_edge_xyz =
      (voxel_face_edge_coef * (voxel_face_edge_pair_xyz.select(1, 1) -
                               voxel_face_edge_pair_xyz.select(1, 0)) +
       voxel_face_edge_pair_xyz.select(1, 0))
          .view({-1, 3, 3});
  auto mask =
      ~voxel_face_edge_xyz.index({"...", 0, 0}).isnan().to(torch::kBool);

  return voxel_face_edge_xyz.index({mask});
}

void tensor_to_mesh(mesh_msgs::MeshGeometry &mesh,
                    mesh_msgs::MeshVertexColors &mesh_color,
                    const torch::Tensor &face_xyz,
                    const torch::Tensor &face_normal_color) {
  auto face_num = face_xyz.size(0);
  mesh.faces.resize(face_num);
  mesh.vertices.resize(face_num * 3);
  mesh_color.vertex_colors.resize(face_num * 3);
  auto face_xyz_a = face_xyz.accessor<float, 3>();
  auto face_normal_color_a = face_normal_color.accessor<float, 2>();
#pragma omp parallel for
  for (int i = 0; i < face_num; i++) {
    for (int k = 0; k < 3; k++) {
      geometry_msgs::Point v;
      v.x = face_xyz_a[i][k][0];
      v.y = face_xyz_a[i][k][1];
      v.z = face_xyz_a[i][k][2];
      mesh.vertices[i * 3 + k] = v;

      std_msgs::ColorRGBA color;
      color.r = face_normal_color_a[i][0];
      color.g = face_normal_color_a[i][1];
      color.b = face_normal_color_a[i][2];
      color.a = 1.0;
      mesh_color.vertex_colors[i * 3 + k] = color;
    }

    mesh_msgs::MeshTriangleIndices face_index;
    face_index.vertex_indices[0] = i * 3;
    face_index.vertex_indices[1] = i * 3 + 1;
    face_index.vertex_indices[2] = i * 3 + 2;
    mesh.faces[i] = face_index;
  }
}

void pub_mesh(const ros::Publisher &_mesh_pub,
              const ros::Publisher &_mesh_color_pub,
              const torch::Tensor &_face_xyz, const std_msgs::Header &_header,
              const std::string &_uuid) {
  if (_mesh_pub.getNumSubscribers() > 0) {
    mesh_msgs::MeshGeometry mesh;
    mesh_msgs::MeshVertexColors mesh_color;
    tensor_to_mesh(mesh, mesh_color, _face_xyz.cpu(),
                   utils::cal_face_normal_color(_face_xyz).cpu());
    mesh_msgs::MeshGeometryStamped mesh_stamped;
    mesh_stamped.header = _header;
    mesh_stamped.uuid = _uuid;
    mesh_stamped.mesh_geometry = mesh;
    _mesh_pub.publish(mesh_stamped);

    mesh_msgs::MeshVertexColorsStamped mesh_color_stamped;
    mesh_color_stamped.header = _header;
    mesh_color_stamped.uuid = _uuid;
    mesh_color_stamped.mesh_vertex_colors = mesh_color;
    _mesh_color_pub.publish(mesh_color_stamped);
  }
}

// stack neighbors in marching cubes rule.
torch::Tensor stack_neighbors(const torch::Tensor &_xyz) {
  return torch::stack(
      {
          _xyz.index({torch::indexing::Slice(0, -1),
                      torch::indexing::Slice(0, -1),
                      torch::indexing::Slice(0, -1)}), // 0
          _xyz.index({torch::indexing::Slice(0, -1), torch::indexing::Slice(1),
                      torch::indexing::Slice(0, -1)}), // 2
          _xyz.index({torch::indexing::Slice(1), torch::indexing::Slice(1),
                      torch::indexing::Slice(0, -1)}), // 6
          _xyz.index({torch::indexing::Slice(1), torch::indexing::Slice(0, -1),
                      torch::indexing::Slice(0, -1)}), // 4
          _xyz.index({torch::indexing::Slice(0, -1),
                      torch::indexing::Slice(0, -1),
                      torch::indexing::Slice(1)}), // 1
          _xyz.index({torch::indexing::Slice(0, -1), torch::indexing::Slice(1),
                      torch::indexing::Slice(1)}), // 3
          _xyz.index({torch::indexing::Slice(1), torch::indexing::Slice(1),
                      torch::indexing::Slice(1)}), // 7
          _xyz.index({torch::indexing::Slice(1), torch::indexing::Slice(0, -1),
                      torch::indexing::Slice(1)}), // 5
      },
      3);
}