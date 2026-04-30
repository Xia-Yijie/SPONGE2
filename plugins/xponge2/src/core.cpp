#include <Python.h>

#include <exception>
#include <set>
#include <stdexcept>
#include <string>

#include "gaff_typing.h"
#include "mol2_reader.h"

namespace XA = Xponge::Assign;

typedef struct
{
    PyObject_HEAD XA::Assignment* assignment;
} PyAssignObject;

static PyObject* PyAssignType = nullptr;

static PyObject* Set_Python_Error_From_Exception()
{
    try
    {
        throw;
    }
    catch (const std::exception& exc)
    {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
    }
    catch (...)
    {
        PyErr_SetString(PyExc_RuntimeError, "unknown xponge2 C++ error");
    }
    return nullptr;
}

template <typename F>
static PyObject* Guarded(F&& f)
{
    try
    {
        return f();
    }
    catch (...)
    {
        return Set_Python_Error_From_Exception();
    }
}

template <typename F>
static int Guarded_Int(F&& f)
{
    try
    {
        return f();
    }
    catch (...)
    {
        Set_Python_Error_From_Exception();
        return -1;
    }
}

static PyObject* New_PyAssign(XA::Assignment&& assignment)
{
    PyObject* object = PyType_GenericAlloc(
        reinterpret_cast<PyTypeObject*>(PyAssignType), 0);
    if (object == nullptr)
    {
        return nullptr;
    }
    PyAssignObject* self = reinterpret_cast<PyAssignObject*>(object);
    try
    {
        self->assignment = new XA::Assignment(std::move(assignment));
    }
    catch (...)
    {
        Py_DECREF(object);
        throw;
    }
    return object;
}

static int PyAssign_init(PyObject* object, PyObject* args, PyObject* kwargs)
{
    PyAssignObject* self = reinterpret_cast<PyAssignObject*>(object);
    const char* name = "ASN";
    static const char* keywords[] = {"name", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|s",
                                     const_cast<char**>(keywords), &name))
    {
        return -1;
    }
    try
    {
        auto* fresh = new XA::Assignment(name);
        delete self->assignment;
        self->assignment = fresh;
    }
    catch (...)
    {
        Set_Python_Error_From_Exception();
        return -1;
    }
    return 0;
}

static void PyAssign_dealloc(PyObject* object)
{
    PyAssignObject* self = reinterpret_cast<PyAssignObject*>(object);
    delete self->assignment;
    PyTypeObject* tp = Py_TYPE(object);
    freefunc tp_free = reinterpret_cast<freefunc>(
        PyType_GetSlot(tp, Py_tp_free));
    tp_free(object);
    Py_DECREF(tp);
}

static XA::Assignment& Require_Assignment(PyAssignObject* self)
{
    if (self->assignment == nullptr)
    {
        throw std::runtime_error("uninitialized Assign object");
    }
    return *self->assignment;
}


static PyObject* PyAssign_atom_numbers(PyAssignObject* self, void*)
{
    return Guarded([&] {
        return PyLong_FromSize_t(Require_Assignment(self).atom_numbers());
    });
}

static PyObject* PyAssign_name(PyAssignObject* self, void*)
{
    return Guarded([&] {
        return PyUnicode_FromString(Require_Assignment(self).name().c_str());
    });
}

static int PyAssign_set_name(PyAssignObject* self, PyObject* value, void*)
{
    if (value == nullptr || !PyUnicode_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "name must be a string");
        return -1;
    }
    PyObject* utf8 = PyUnicode_AsUTF8String(value);
    if (utf8 == nullptr)
    {
        return -1;
    }
    const char* name = PyBytes_AsString(utf8);
    if (name == nullptr)
    {
        Py_DECREF(utf8);
        return -1;
    }
    const int rc = Guarded_Int([&] {
        Require_Assignment(self).set_name(name);
        return 0;
    });
    Py_DECREF(utf8);
    return rc;
}

template <typename F>
static PyObject* List_From_Index(std::size_t n, F&& to_py)
{
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(n));
    if (list == nullptr)
    {
        return nullptr;
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        PyObject* value = to_py(i);
        if (value == nullptr)
        {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SetItem(list, static_cast<Py_ssize_t>(i), value);
    }
    return list;
}

static int Dict_Set_Steal_Pair(PyObject* dict, PyObject* key, PyObject* value)
{
    if (key == nullptr || value == nullptr)
    {
        Py_XDECREF(key);
        Py_XDECREF(value);
        return -1;
    }
    const int rc = PyDict_SetItem(dict, key, value);
    Py_DECREF(key);
    Py_DECREF(value);
    return rc;
}

template <typename F>
static PyObject* Index_Keyed_Dict(std::size_t n, F&& value_to_py)
{
    PyObject* result = PyDict_New();
    if (result == nullptr)
    {
        return nullptr;
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        if (Dict_Set_Steal_Pair(result, PyLong_FromSize_t(i),
                                value_to_py(i)) < 0)
        {
            Py_DECREF(result);
            return nullptr;
        }
    }
    return result;
}

static PyObject* PyAssign_atoms(PyAssignObject* self, void*)
{
    return Guarded([&] {
        const auto& atoms = Require_Assignment(self).atoms();
        return List_From_Index(atoms.size(), [&](std::size_t i) {
            return PyUnicode_FromString(atoms[i].element.c_str());
        });
    });
}

static PyObject* PyAssign_names(PyAssignObject* self, void*)
{
    return Guarded([&] {
        const auto& atoms = Require_Assignment(self).atoms();
        return List_From_Index(atoms.size(), [&](std::size_t i) {
            return PyUnicode_FromString(atoms[i].name.c_str());
        });
    });
}

static PyObject* PyAssign_element_details(PyAssignObject* self, void*)
{
    return Guarded([&] {
        const auto& atoms = Require_Assignment(self).atoms();
        return List_From_Index(atoms.size(), [&](std::size_t i) {
            return PyUnicode_FromString(atoms[i].element_detail.c_str());
        });
    });
}

static PyObject* PyAssign_charge(PyAssignObject* self, void*)
{
    return Guarded([&] {
        const auto& atoms = Require_Assignment(self).atoms();
        return List_From_Index(atoms.size(), [&](std::size_t i) {
            return PyFloat_FromDouble(atoms[i].charge);
        });
    });
}

static PyObject* PyAssign_formal_charge(PyAssignObject* self, void*)
{
    return Guarded([&] {
        const auto& atoms = Require_Assignment(self).atoms();
        return List_From_Index(atoms.size(), [&](std::size_t i) {
            return PyLong_FromLong(atoms[i].formal_charge);
        });
    });
}

static PyObject* PyAssign_coordinate(PyAssignObject* self, void*)
{
    return Guarded([&] {
        const auto& atoms = Require_Assignment(self).atoms();
        return List_From_Index(atoms.size(), [&](std::size_t i) {
            const auto& c = atoms[i].coordinate;
            return Py_BuildValue("(ddd)", c.x, c.y, c.z);
        });
    });
}

static PyObject* PyAssign_bonds(PyAssignObject* self, void*)
{
    return Guarded([&] {
        const auto& bonds = Require_Assignment(self).bonds();
        return Index_Keyed_Dict(bonds.size(), [&](std::size_t i) -> PyObject* {
            PyObject* inner = PyDict_New();
            if (inner == nullptr)
            {
                return nullptr;
            }
            for (const auto& item : bonds[i])
            {
                if (Dict_Set_Steal_Pair(inner, PyLong_FromLong(item.first),
                                        PyLong_FromLong(item.second)) < 0)
                {
                    Py_DECREF(inner);
                    return nullptr;
                }
            }
            return inner;
        });
    });
}

static PyObject* PyAssign_atom_marker(PyAssignObject* self, void*)
{
    return Guarded([&] {
        const auto& markers = Require_Assignment(self).atom_markers();
        return Index_Keyed_Dict(markers.size(), [&](std::size_t i) -> PyObject* {
            PyObject* inner = PyDict_New();
            if (inner == nullptr)
            {
                return nullptr;
            }
            for (const auto& item : markers[i])
            {
                if (Dict_Set_Steal_Pair(inner,
                                        PyUnicode_FromString(item.first.c_str()),
                                        PyLong_FromLong(item.second)) < 0)
                {
                    Py_DECREF(inner);
                    return nullptr;
                }
            }
            return inner;
        });
    });
}

static PyObject* String_Set_From_Set(const std::set<std::string>& markers)
{
    PyObject* result = PySet_New(nullptr);
    if (result == nullptr)
    {
        return nullptr;
    }
    for (const auto& marker : markers)
    {
        PyObject* value = PyUnicode_FromString(marker.c_str());
        if (value == nullptr || PySet_Add(result, value) < 0)
        {
            Py_XDECREF(value);
            Py_DECREF(result);
            return nullptr;
        }
        Py_DECREF(value);
    }
    return result;
}

static PyObject* PyAssign_bond_marker(PyAssignObject* self, void*)
{
    return Guarded([&] {
        const auto& markers = Require_Assignment(self).bond_markers();
        return Index_Keyed_Dict(markers.size(), [&](std::size_t i) -> PyObject* {
            PyObject* inner = PyDict_New();
            if (inner == nullptr)
            {
                return nullptr;
            }
            for (const auto& item : markers[i])
            {
                if (Dict_Set_Steal_Pair(inner, PyLong_FromLong(item.first),
                                        String_Set_From_Set(item.second)) < 0)
                {
                    Py_DECREF(inner);
                    return nullptr;
                }
            }
            return inner;
        });
    });
}

static PyObject* PyAssign_atom_types(PyAssignObject* self, void*)
{
    return Guarded([&] {
        const auto& atom_types = Require_Assignment(self).atom_types();
        return Index_Keyed_Dict(atom_types.size(), [&](std::size_t i) {
            return atom_types[i].empty()
                       ? Py_NewRef(Py_None)
                       : PyUnicode_FromString(atom_types[i].c_str());
        });
    });
}

static PyObject* PyAssign_built(PyAssignObject* self, void*)
{
    return Guarded([&]() -> PyObject* {
        if (Require_Assignment(self).built())
        {
            Py_RETURN_TRUE;
        }
        Py_RETURN_FALSE;
    });
}

static int PyAssign_set_built(PyAssignObject* self, PyObject* value, void*)
{
    if (value == nullptr)
    {
        PyErr_SetString(PyExc_TypeError, "cannot delete built");
        return -1;
    }
    return Guarded_Int([&] {
        Require_Assignment(self).set_built(PyObject_IsTrue(value) == 1);
        return 0;
    });
}

static PyObject* PyAssign_kekulized(PyAssignObject* self, void*)
{
    return Guarded([&]() -> PyObject* {
        if (Require_Assignment(self).kekulized())
        {
            Py_RETURN_TRUE;
        }
        Py_RETURN_FALSE;
    });
}

static int PyAssign_set_kekulized(PyAssignObject* self,
                                  PyObject* value,
                                  void*)
{
    if (value == nullptr)
    {
        PyErr_SetString(PyExc_TypeError, "cannot delete kekulized");
        return -1;
    }
    return Guarded_Int([&] {
        Require_Assignment(self).set_kekulized(PyObject_IsTrue(value) == 1);
        return 0;
    });
}

static PyObject* PyAssign_add_atom(PyAssignObject* self,
                                   PyObject* args,
                                   PyObject* kwargs)
{
    const char* element = nullptr;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    const char* name = "";
    double charge = 0.0;
    static const char* keywords[] = {
        "element", "x", "y", "z", "name", "charge", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sddd|sd",
                                     const_cast<char**>(keywords), &element,
                                     &x, &y, &z, &name, &charge))
    {
        return nullptr;
    }
    return Guarded([&]() -> PyObject* {
        Require_Assignment(self).Add_Atom(element, x, y, z, name, charge);
        Py_RETURN_NONE;
    });
}

static PyObject* PyAssign_add_bond(PyAssignObject* self,
                                   PyObject* args,
                                   PyObject* kwargs)
{
    int atom1 = 0;
    int atom2 = 0;
    int order = -1;
    static const char* keywords[] = {"atom1", "atom2", "order", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|i",
                                     const_cast<char**>(keywords), &atom1,
                                     &atom2, &order))
    {
        return nullptr;
    }
    return Guarded([&]() -> PyObject* {
        Require_Assignment(self).Add_Bond(atom1, atom2, order);
        Py_RETURN_NONE;
    });
}

static PyObject* PyAssign_add_atom_marker(PyAssignObject* self,
                                          PyObject* args,
                                          PyObject* kwargs)
{
    int atom = 0;
    const char* marker = nullptr;
    static const char* keywords[] = {"atom", "marker", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "is",
                                     const_cast<char**>(keywords), &atom,
                                     &marker))
    {
        return nullptr;
    }
    return Guarded([&]() -> PyObject* {
        Require_Assignment(self).Add_Atom_Marker(atom, marker);
        Py_RETURN_NONE;
    });
}

static PyObject* PyAssign_add_bond_marker(PyAssignObject* self,
                                          PyObject* args,
                                          PyObject* kwargs)
{
    int atom1 = 0;
    int atom2 = 0;
    const char* marker = nullptr;
    int only1 = 0;
    static const char* keywords[] = {
        "atom1", "atom2", "marker", "only1", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iis|p",
                                     const_cast<char**>(keywords), &atom1,
                                     &atom2, &marker, &only1))
    {
        return nullptr;
    }
    return Guarded([&]() -> PyObject* {
        Require_Assignment(self).Add_Bond_Marker(atom1, atom2, marker,
                                                 only1 != 0);
        Py_RETURN_NONE;
    });
}

static PyObject* PyAssign_delete_bond(PyAssignObject* self,
                                      PyObject* args,
                                      PyObject* kwargs)
{
    int atom1 = 0;
    int atom2 = 0;
    static const char* keywords[] = {"atom1", "atom2", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii",
                                     const_cast<char**>(keywords), &atom1,
                                     &atom2))
    {
        return nullptr;
    }
    return Guarded([&]() -> PyObject* {
        Require_Assignment(self).Delete_Bond(atom1, atom2);
        Py_RETURN_NONE;
    });
}

static PyObject* PyAssign_check_connectivity(PyAssignObject* self, PyObject*)
{
    return Guarded([&]() -> PyObject* {
        if (Require_Assignment(self).Check_Connectivity())
        {
            Py_RETURN_TRUE;
        }
        Py_RETURN_FALSE;
    });
}

static PyObject* PyAssign_determine_atom_type(PyAssignObject* self,
                                              PyObject* args)
{
    const char* rule = nullptr;
    if (!PyArg_ParseTuple(args, "s", &rule))
    {
        return nullptr;
    }
    if (std::string(rule) != "gaff")
    {
        PyErr_SetString(PyExc_ValueError,
                        "only the 'gaff' atom type rule is recognized");
        return nullptr;
    }
    return Guarded([&]() -> PyObject* {
        XA::Determine_Gaff_Atom_Type(Require_Assignment(self));
        Py_RETURN_NONE;
    });
}

static bool Atom_Types_Missing(const XA::Assignment& assignment)
{
    const auto& atom_types = assignment.atom_types();
    if (atom_types.size() < assignment.atom_numbers())
    {
        return true;
    }
    for (std::size_t i = 0; i < assignment.atom_numbers(); ++i)
    {
        if (atom_types[i].empty())
        {
            return true;
        }
    }
    return false;
}

static PyObject* PyAssign_save_as_mol2(PyAssignObject* self,
                                       PyObject* args,
                                       PyObject* kwargs)
{
    const char* filename = nullptr;
    const char* atomtype = "sybyl";
    static const char* keywords[] = {"filename", "atomtype", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|s",
                                     const_cast<char**>(keywords), &filename,
                                     &atomtype))
    {
        return nullptr;
    }
    return Guarded([&]() -> PyObject* {
        XA::Assignment& assignment = Require_Assignment(self);
        if (std::string(atomtype) == "gaff" && Atom_Types_Missing(assignment))
        {
            XA::Determine_Gaff_Atom_Type(assignment);
        }
        XA::Save_Assignment_As_Mol2(assignment, filename, atomtype);
        Py_RETURN_NONE;
    });
}

PyDoc_STRVAR(PyAssign_add_atom_doc,
             "add_atom($self, element, x, y, z, name='', charge=0.0, /)\n"
             "--\n\nAdd an atom to the assignment.");
PyDoc_STRVAR(PyAssign_add_bond_doc,
             "add_bond($self, atom1, atom2, order=-1, /)\n"
             "--\n\nAdd a bond to the assignment.");
PyDoc_STRVAR(PyAssign_add_atom_marker_doc,
             "add_atom_marker($self, atom, marker, /)\n"
             "--\n\nAdd a marker to an atom.");
PyDoc_STRVAR(PyAssign_add_bond_marker_doc,
             "add_bond_marker($self, atom1, atom2, marker, only1=False, /)\n"
             "--\n\nAdd a marker to a bond.");
PyDoc_STRVAR(PyAssign_delete_bond_doc,
             "delete_bond($self, atom1, atom2, /)\n"
             "--\n\nDelete a bond from the assignment.");
PyDoc_STRVAR(PyAssign_check_connectivity_doc,
             "check_connectivity($self, /)\n"
             "--\n\nReturn whether all atoms are in one connected graph.");
PyDoc_STRVAR(PyAssign_determine_atom_type_doc,
             "determine_atom_type($self, rule, /)\n"
             "--\n\nDetermine atom types.");
PyDoc_STRVAR(PyAssign_save_as_mol2_doc,
             "save_as_mol2($self, filename, atomtype='sybyl', /)\n"
             "--\n\nSave the assignment as a mol2 file.");

static PyMethodDef PyAssign_methods[] = {
    {"add_atom", reinterpret_cast<PyCFunction>(PyAssign_add_atom),
     METH_VARARGS | METH_KEYWORDS, PyAssign_add_atom_doc},
    {"Add_Atom", reinterpret_cast<PyCFunction>(PyAssign_add_atom),
     METH_VARARGS | METH_KEYWORDS, PyAssign_add_atom_doc},
    {"add_bond", reinterpret_cast<PyCFunction>(PyAssign_add_bond),
     METH_VARARGS | METH_KEYWORDS, PyAssign_add_bond_doc},
    {"Add_Bond", reinterpret_cast<PyCFunction>(PyAssign_add_bond),
     METH_VARARGS | METH_KEYWORDS, PyAssign_add_bond_doc},
    {"add_atom_marker",
     reinterpret_cast<PyCFunction>(PyAssign_add_atom_marker),
     METH_VARARGS | METH_KEYWORDS, PyAssign_add_atom_marker_doc},
    {"Add_Atom_Marker",
     reinterpret_cast<PyCFunction>(PyAssign_add_atom_marker),
     METH_VARARGS | METH_KEYWORDS, PyAssign_add_atom_marker_doc},
    {"add_bond_marker",
     reinterpret_cast<PyCFunction>(PyAssign_add_bond_marker),
     METH_VARARGS | METH_KEYWORDS, PyAssign_add_bond_marker_doc},
    {"Add_Bond_Marker",
     reinterpret_cast<PyCFunction>(PyAssign_add_bond_marker),
     METH_VARARGS | METH_KEYWORDS, PyAssign_add_bond_marker_doc},
    {"delete_bond", reinterpret_cast<PyCFunction>(PyAssign_delete_bond),
     METH_VARARGS | METH_KEYWORDS, PyAssign_delete_bond_doc},
    {"Delete_Bond", reinterpret_cast<PyCFunction>(PyAssign_delete_bond),
     METH_VARARGS | METH_KEYWORDS, PyAssign_delete_bond_doc},
    {"check_connectivity",
     reinterpret_cast<PyCFunction>(PyAssign_check_connectivity), METH_NOARGS,
     PyAssign_check_connectivity_doc},
    {"Check_Connectivity",
     reinterpret_cast<PyCFunction>(PyAssign_check_connectivity), METH_NOARGS,
     PyAssign_check_connectivity_doc},
    {"determine_atom_type",
     reinterpret_cast<PyCFunction>(PyAssign_determine_atom_type),
     METH_VARARGS, PyAssign_determine_atom_type_doc},
    {"Determine_Atom_Type",
     reinterpret_cast<PyCFunction>(PyAssign_determine_atom_type),
     METH_VARARGS, PyAssign_determine_atom_type_doc},
    {"save_as_mol2", reinterpret_cast<PyCFunction>(PyAssign_save_as_mol2),
     METH_VARARGS | METH_KEYWORDS, PyAssign_save_as_mol2_doc},
    {"Save_As_Mol2", reinterpret_cast<PyCFunction>(PyAssign_save_as_mol2),
     METH_VARARGS | METH_KEYWORDS, PyAssign_save_as_mol2_doc},
    {nullptr, nullptr, 0, nullptr}};

static PyGetSetDef PyAssign_getset[] = {
    {"name", reinterpret_cast<getter>(PyAssign_name),
     reinterpret_cast<setter>(PyAssign_set_name), nullptr, nullptr},
    {"atom_numbers", reinterpret_cast<getter>(PyAssign_atom_numbers), nullptr,
     nullptr, nullptr},
    {"atoms", reinterpret_cast<getter>(PyAssign_atoms), nullptr, nullptr,
     nullptr},
    {"names", reinterpret_cast<getter>(PyAssign_names), nullptr, nullptr,
     nullptr},
    {"element_details", reinterpret_cast<getter>(PyAssign_element_details),
     nullptr, nullptr, nullptr},
    {"coordinate", reinterpret_cast<getter>(PyAssign_coordinate), nullptr,
     nullptr, nullptr},
    {"charge", reinterpret_cast<getter>(PyAssign_charge), nullptr, nullptr,
     nullptr},
    {"formal_charge", reinterpret_cast<getter>(PyAssign_formal_charge),
     nullptr, nullptr, nullptr},
    {"bonds", reinterpret_cast<getter>(PyAssign_bonds), nullptr, nullptr,
     nullptr},
    {"atom_marker", reinterpret_cast<getter>(PyAssign_atom_marker), nullptr,
     nullptr, nullptr},
    {"bond_marker", reinterpret_cast<getter>(PyAssign_bond_marker), nullptr,
     nullptr, nullptr},
    {"atom_types", reinterpret_cast<getter>(PyAssign_atom_types), nullptr,
     nullptr, nullptr},
    {"built", reinterpret_cast<getter>(PyAssign_built),
     reinterpret_cast<setter>(PyAssign_set_built), nullptr, nullptr},
    {"kekulized", reinterpret_cast<getter>(PyAssign_kekulized),
     reinterpret_cast<setter>(PyAssign_set_kekulized), nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

static PyType_Slot PyAssignTypeSlots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(PyAssign_dealloc)},
    {Py_tp_doc, const_cast<char*>(
                    "Assign(name='ASN')\n--\n\nXponge assignment object.")},
    {Py_tp_methods, PyAssign_methods},
    {Py_tp_getset, PyAssign_getset},
    {Py_tp_init, reinterpret_cast<void*>(PyAssign_init)},
    {Py_tp_new, reinterpret_cast<void*>(PyType_GenericNew)},
    {0, nullptr}};

static PyType_Spec PyAssignTypeSpec = {
    "xponge2.Assign",
    sizeof(PyAssignObject),
    0,
    Py_TPFLAGS_DEFAULT,
    PyAssignTypeSlots};

static PyObject* Module_get_assignment_from_mol2(PyObject*,
                                                 PyObject* args,
                                                 PyObject* kwargs)
{
    const char* filename = nullptr;
    PyObject* total_charge = Py_None;
    static const char* keywords[] = {"file", "total_charge", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|O",
                                     const_cast<char**>(keywords), &filename,
                                     &total_charge))
    {
        return nullptr;
    }
    if (total_charge != Py_None)
    {
        PyErr_SetString(PyExc_NotImplementedError,
                        "total_charge is not yet supported by xponge2");
        return nullptr;
    }
    return Guarded([&] {
        return New_PyAssign(XA::Get_Assignment_From_Mol2(filename));
    });
}

PyDoc_STRVAR(Module_get_assignment_from_mol2_doc,
             "get_assignment_from_mol2(file, total_charge=None)\n"
             "--\n\nRead a mol2 file as an Assign object.");

static PyMethodDef Xponge2Methods[] = {
    {"get_assignment_from_mol2",
     reinterpret_cast<PyCFunction>(Module_get_assignment_from_mol2),
     METH_VARARGS | METH_KEYWORDS, Module_get_assignment_from_mol2_doc},
    {"Get_Assignment_From_Mol2",
     reinterpret_cast<PyCFunction>(Module_get_assignment_from_mol2),
     METH_VARARGS | METH_KEYWORDS, Module_get_assignment_from_mol2_doc},
    {nullptr, nullptr, 0, nullptr}};

static PyModuleDef Xponge2Module = {
    PyModuleDef_HEAD_INIT,
    "xponge2._core",
    nullptr,
    -1,
    Xponge2Methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

PyMODINIT_FUNC PyInit__core(void)
{
    PyObject* module = PyModule_Create(&Xponge2Module);
    if (module == nullptr)
    {
        return nullptr;
    }

    PyAssignType = PyType_FromSpec(&PyAssignTypeSpec);
    if (PyAssignType == nullptr)
    {
        Py_DECREF(module);
        return nullptr;
    }

    if (PyModule_AddObject(module, "Assign", Py_NewRef(PyAssignType)) < 0)
    {
        Py_DECREF(PyAssignType);
        Py_DECREF(module);
        return nullptr;
    }
    PyModule_AddStringConstant(module, "__version__", XPONGE2_VERSION);
    return module;
}
