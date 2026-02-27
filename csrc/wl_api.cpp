/*
 * wl_api.cpp -- C ABI adapter for xLearn WASM port
 *
 * Wraps xLearn's C API for use from JavaScript via Emscripten.
 * Adds:
 *   - CSR DMatrix construction (not in upstream C API)
 *   - MEMFS-based model byte I/O (fit returns bytes, predict takes bytes)
 *   - Safe prediction output (copies to caller buffer)
 *
 * Compile with: emcc csrc/wl_api.cpp + upstream sources
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <unistd.h>
#include <fcntl.h>

#include "src/c_api/c_api.h"
#include "src/c_api/c_api_error.h"
#include "src/data/data_structure.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- error handling ---------- */

static char last_error[1024] = "";

static void set_error(const char *msg) {
  strncpy(last_error, msg, sizeof(last_error) - 1);
  last_error[sizeof(last_error) - 1] = '\0';
}

const char* wl_xl_get_last_error(void) {
  return last_error;
}

/* ---------- handle lifecycle ---------- */

int wl_xl_create(const char *model_type, void **out) {
  last_error[0] = '\0';
  if (!model_type || !out) {
    set_error("wl_xl_create: null argument");
    return -1;
  }
  XL handle = nullptr;
  int ret = XLearnCreate(model_type, &handle);
  if (ret != 0) {
    const char *err = XLearnGetLastError();
    set_error(err ? err : "XLearnCreate failed");
    return -1;
  }
  /* Set WASM-friendly defaults */
  XLearnSetBool(&handle, "quiet", true);
  XLearnSetBool(&handle, "lock_free", false);
  XLearnSetBool(&handle, "from_file", false);
  XLearnSetBool(&handle, "bin_out", false);
  XLearnSetBool(&handle, "early_stop", false);
  XLearnSetInt(&handle, "nthread", 1);
  XLearnSetStr(&handle, "log", "/dev/null");
  *out = handle;
  return 0;
}

void wl_xl_free_handle(void *handle) {
  if (handle) {
    XLearn *xl = reinterpret_cast<XLearn*>(handle);
    delete xl;
  }
}

/* ---------- parameter setters ---------- */

int wl_xl_set_str(void *handle, const char *key, const char *value) {
  if (!handle || !key || !value) {
    set_error("wl_xl_set_str: null argument");
    return -1;
  }
  return XLearnSetStr(&handle, key, value);
}

int wl_xl_set_int(void *handle, const char *key, int value) {
  if (!handle || !key) {
    set_error("wl_xl_set_int: null argument");
    return -1;
  }
  return XLearnSetInt(&handle, key, value);
}

int wl_xl_set_float(void *handle, const char *key, float value) {
  if (!handle || !key) {
    set_error("wl_xl_set_float: null argument");
    return -1;
  }
  return XLearnSetFloat(&handle, key, value);
}

int wl_xl_set_bool(void *handle, const char *key, int value) {
  if (!handle || !key) {
    set_error("wl_xl_set_bool: null argument");
    return -1;
  }
  return XLearnSetBool(&handle, key, (bool)value);
}

/* ---------- DMatrix from dense array ---------- */

int wl_xl_create_dmatrix_dense(
    const float *data, int nrow, int ncol,
    const float *label,
    const int *field_map,
    void **out
) {
  last_error[0] = '\0';
  if (!data || nrow <= 0 || ncol <= 0 || !out) {
    set_error("wl_xl_create_dmatrix_dense: invalid arguments");
    return -1;
  }

  try {
    xLearn::DMatrix *matrix = new xLearn::DMatrix();
    matrix->has_label = (label != nullptr);

    for (int i = 0; i < nrow; ++i) {
      matrix->AddRow();
      if (label) {
        matrix->Y[i] = label[i];
      }
      float norm = 0.0f;
      for (int j = 0; j < ncol; ++j) {
        float val = data[i * ncol + j];
        if (val == 0.0f) continue;  /* skip zeros (match file-reader) */
        xLearn::index_t field_id = field_map ? (xLearn::index_t)field_map[j] : 0;
        matrix->AddNode(i, (xLearn::index_t)j, val, field_id);
        norm += val * val;
      }
      matrix->norm[i] = (norm > 0.0f) ? (1.0f / norm) : 1.0f;
    }

    *out = matrix;
    return 0;
  } catch (const std::exception &e) {
    set_error(e.what());
    return -1;
  }
}

/* ---------- DMatrix from CSR sparse arrays ---------- */

int wl_xl_create_dmatrix_csr(
    const float *values, int nnz,
    const int *col_indices,
    const int *row_ptr, int nrow,
    int ncol,
    const float *label,
    const int *field_map,
    void **out
) {
  last_error[0] = '\0';
  if (!values || !col_indices || !row_ptr || nrow <= 0 || ncol <= 0 || !out) {
    set_error("wl_xl_create_dmatrix_csr: invalid arguments");
    return -1;
  }

  try {
    xLearn::DMatrix *matrix = new xLearn::DMatrix();
    matrix->has_label = (label != nullptr);

    for (int i = 0; i < nrow; ++i) {
      matrix->AddRow();
      if (label) {
        matrix->Y[i] = label[i];
      }
      float norm = 0.0f;
      int start = row_ptr[i];
      int end = row_ptr[i + 1];
      for (int j = start; j < end; ++j) {
        int col = col_indices[j];
        float val = values[j];
        xLearn::index_t field_id = field_map ? (xLearn::index_t)field_map[col] : 0;
        matrix->AddNode(i, (xLearn::index_t)col, val, field_id);
        norm += val * val;
      }
      matrix->norm[i] = (norm > 0.0f) ? (1.0f / norm) : 1.0f;
    }

    *out = matrix;
    return 0;
  } catch (const std::exception &e) {
    set_error(e.what());
    return -1;
  }
}

void wl_xl_free_dmatrix(void *dmatrix) {
  if (dmatrix) {
    xLearn::DMatrix *dm = reinterpret_cast<xLearn::DMatrix*>(dmatrix);
    dm->Reset();
    delete dm;
  }
}

/* ---------- stdout suppression ---------- */

static int saved_stdout_fd = -1;

static void suppress_stdout(void) {
  fflush(stdout);
  saved_stdout_fd = dup(1);
  int devnull = open("/dev/null", O_WRONLY);
  dup2(devnull, 1);
  close(devnull);
}

static void restore_stdout(void) {
  if (saved_stdout_fd >= 0) {
    fflush(stdout);
    dup2(saved_stdout_fd, 1);
    close(saved_stdout_fd);
    saved_stdout_fd = -1;
  }
}

/* ---------- train ---------- */

static int fit_counter = 0;

int wl_xl_fit(
    void *handle,
    void *dtrain,
    void *dvalid,
    char **out_model_buf,
    int *out_model_len
) {
  last_error[0] = '\0';
  if (!handle || !dtrain || !out_model_buf || !out_model_len) {
    set_error("wl_xl_fit: null argument");
    return -1;
  }

  /* Assign DMatrix to handle */
  DataHandle train_dh = dtrain;
  int ret = XLearnSetDMatrix(&handle, "train", &train_dh);
  if (ret != 0) {
    const char *err = XLearnGetLastError();
    set_error(err ? err : "XLearnSetDMatrix(train) failed");
    return -1;
  }

  if (dvalid) {
    DataHandle valid_dh = dvalid;
    ret = XLearnSetDMatrix(&handle, "validate", &valid_dh);
    if (ret != 0) {
      const char *err = XLearnGetLastError();
      set_error(err ? err : "XLearnSetDMatrix(validate) failed");
      return -1;
    }
  }

  /* Train and save model to MEMFS */
  char model_path[64];
  snprintf(model_path, sizeof(model_path), "/tmp/wl_xl_model_%d", fit_counter++);

  suppress_stdout();
  ret = XLearnFit(&handle, model_path);
  restore_stdout();
  if (ret != 0) {
    const char *err = XLearnGetLastError();
    set_error(err ? err : "XLearnFit failed");
    remove(model_path);
    return -1;
  }

  /* Read model bytes from MEMFS */
  FILE *f = fopen(model_path, "rb");
  if (!f) {
    set_error("wl_xl_fit: cannot read model file from MEMFS");
    return -1;
  }

  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);

  char *buf = (char *)malloc((size_t)size);
  if (!buf) {
    fclose(f);
    remove(model_path);
    set_error("wl_xl_fit: allocation failed");
    return -1;
  }

  fread(buf, 1, (size_t)size, f);
  fclose(f);
  remove(model_path);

  *out_model_buf = buf;
  *out_model_len = (int)size;
  return 0;
}

/* ---------- predict ---------- */

int wl_xl_predict(
    void *handle,
    const char *model_buf, int model_len,
    void *dtest,
    float **out_preds, int *out_len
) {
  last_error[0] = '\0';
  if (!handle || !model_buf || model_len <= 0 || !dtest || !out_preds || !out_len) {
    set_error("wl_xl_predict: null argument");
    return -1;
  }

  /* Write model bytes to MEMFS */
  char model_path[64];
  snprintf(model_path, sizeof(model_path), "/tmp/wl_xl_pred_%d", fit_counter++);

  FILE *f = fopen(model_path, "wb");
  if (!f) {
    set_error("wl_xl_predict: cannot create model file in MEMFS");
    return -1;
  }
  fwrite(model_buf, 1, (size_t)model_len, f);
  fclose(f);

  /* Assign test DMatrix */
  DataHandle test_dh = dtest;
  int ret = XLearnSetDMatrix(&handle, "test", &test_dh);
  if (ret != 0) {
    remove(model_path);
    const char *err = XLearnGetLastError();
    set_error(err ? err : "XLearnSetDMatrix(test) failed");
    return -1;
  }

  /* Predict */
  uint64_t length = 0;
  const float *arr = nullptr;
  suppress_stdout();
  ret = XLearnPredictForMat(&handle, model_path, &length, &arr);
  restore_stdout();
  remove(model_path);

  if (ret != 0) {
    const char *err = XLearnGetLastError();
    set_error(err ? err : "XLearnPredictForMat failed");
    return -1;
  }

  /* Copy predictions to caller-owned buffer (arr points to static vector) */
  int n = (int)length;
  float *result = (float *)malloc((size_t)n * sizeof(float));
  if (!result) {
    set_error("wl_xl_predict: allocation failed");
    return -1;
  }
  memcpy(result, arr, (size_t)n * sizeof(float));

  *out_preds = result;
  *out_len = n;
  return 0;
}

/* ---------- memory management ---------- */

void wl_xl_free_buffer(void *ptr) {
  free(ptr);
}

#ifdef __cplusplus
}
#endif
