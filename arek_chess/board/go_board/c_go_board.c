#define Py_LIMITED_API
#include <Python.h>
#include "go_board.h"

PyObject *sum_wrapper(PyObject *obj, PyObject *args) {
    const *char a, b;

    if (!PyArg_ParseTuple(args, "LL", &a, &b))
        return NULL;

    return PyLong_FromLong(sum(a, b));
}

static PyMethodDef BoardMethods[] = {
    {"sum", sum, METH_VARARGS, "Add two numbers."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef go_board = {
    PyModuleDef_HEAD_INIT, "go_board", NULL, -1, BoardMethods
};

PyMODINIT_FUNC PyInit_go_board(void) {
    return PyModule_Create(&go_board);
}