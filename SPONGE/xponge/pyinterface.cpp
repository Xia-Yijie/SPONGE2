#include <Python.h>

#include <exception>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "gaff_typing.h"
#include "model.h"
#include "mol2_reader.h"
#include "parmchk2.h"
#include "parameters.h"
#include "sponge_writer.h"

namespace XA = Xponge::Assign;
namespace Amber = Xponge::Amber;

typedef struct
{
    PyObject_HEAD XA::Assignment* assignment;
} PyAssignObject;

typedef struct
{
    PyObject_HEAD Amber::GaffParameters* parameters;
} PyGaffParametersObject;

static PyObject* PyAssignType = nullptr;
static PyObject* PyGaffParametersType = nullptr;

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

static PyObject* New_PyGaffParameters(Amber::GaffParameters&& parameters)
{
    PyObject* object = PyType_GenericAlloc(
        reinterpret_cast<PyTypeObject*>(PyGaffParametersType), 0);
    if (object == nullptr)
    {
        return nullptr;
    }
    PyGaffParametersObject* self =
        reinterpret_cast<PyGaffParametersObject*>(object);
    try
    {
        self->parameters =
            new Amber::GaffParameters(std::move(parameters));
    }
    catch (...)
    {
        Py_DECREF(object);
        throw;
    }
    return object;
}

static void PyGaffParameters_dealloc(PyObject* object)
{
    PyGaffParametersObject* self =
        reinterpret_cast<PyGaffParametersObject*>(object);
    delete self->parameters;
    PyTypeObject* tp = Py_TYPE(object);
    freefunc tp_free = reinterpret_cast<freefunc>(
        PyType_GetSlot(tp, Py_tp_free));
    tp_free(object);
    Py_DECREF(tp);
}

static Amber::GaffParameters& Require_Gaff_Parameters(
    PyGaffParametersObject* self)
{
    if (self->parameters == nullptr)
    {
        throw std::runtime_error("uninitialized GaffParameters object");
    }
    return *self->parameters;
}

static PyType_Slot PyGaffParametersTypeSlots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(PyGaffParameters_dealloc)},
    {Py_tp_doc, const_cast<char*>("Native GAFF parameter table.")},
    {0, nullptr}};

static PyType_Spec PyGaffParametersTypeSpec = {
    "xponge2.GaffParameters",
    sizeof(PyGaffParametersObject),
    0,
    Py_TPFLAGS_DEFAULT,
    PyGaffParametersTypeSlots};

static PyObject* Get_Attr(PyObject* object, const char* name)
{
    PyObject* value = PyObject_GetAttrString(object, name);
    if (value == nullptr)
    {
        throw std::runtime_error(std::string("missing attribute ") + name);
    }
    return value;
}

static bool Has_Attr(PyObject* object, const char* name)
{
    const int result = PyObject_HasAttrString(object, name);
    return result == 1;
}

static std::string Object_To_String(PyObject* object)
{
    PyObject* utf8 = PyUnicode_AsUTF8String(object);
    if (utf8 == nullptr)
    {
        throw std::runtime_error("expected string");
    }
    const char* text = PyBytes_AsString(utf8);
    if (text == nullptr)
    {
        Py_DECREF(utf8);
        throw std::runtime_error("expected string");
    }
    std::string result = text;
    Py_DECREF(utf8);
    return result;
}

static std::string String_Attr(PyObject* object,
                               const char* name,
                               const std::string& fallback = "")
{
    PyObject* value = PyObject_GetAttrString(object, name);
    if (value == nullptr)
    {
        PyErr_Clear();
        return fallback;
    }
    if (value == Py_None)
    {
        Py_DECREF(value);
        return fallback;
    }
    std::string result = Object_To_String(value);
    Py_DECREF(value);
    return result;
}

static double Double_Attr(PyObject* object, const char* name)
{
    PyObject* value = Get_Attr(object, name);
    const double result = PyFloat_AsDouble(value);
    Py_DECREF(value);
    if (PyErr_Occurred())
    {
        throw std::runtime_error(std::string("expected float attribute ") +
                                 name);
    }
    return result;
}

static std::vector<PyObject*> Iter_Items(PyObject* iterable)
{
    std::vector<PyObject*> items;
    PyObject* iterator = PyObject_GetIter(iterable);
    if (iterator == nullptr)
    {
        throw std::runtime_error("object is not iterable");
    }
    while (PyObject* item = PyIter_Next(iterator))
    {
        items.push_back(item);
    }
    Py_DECREF(iterator);
    if (PyErr_Occurred())
    {
        for (PyObject* item : items)
        {
            Py_DECREF(item);
        }
        throw std::runtime_error("failed to iterate Python object");
    }
    return items;
}

static Xponge::Molecule Molecule_From_Python(PyObject* object)
{
    Xponge::Molecule molecule;
    PyObject* residues_object = Get_Attr(object, "residues");
    auto residues = Iter_Items(residues_object);
    Py_DECREF(residues_object);
    try
    {
        for (PyObject* py_residue : residues)
        {
            Xponge::ResidueType residue;
            residue.name = String_Attr(py_residue, "name", "MOL");
            residue.forcefield = String_Attr(py_residue, "forcefield", "");
            residue.head = String_Attr(py_residue, "head", "");
            residue.tail = String_Attr(py_residue, "tail", "");

            PyObject* atoms_object = Get_Attr(py_residue, "atoms");
            auto atoms = Iter_Items(atoms_object);
            Py_DECREF(atoms_object);
            for (PyObject* py_atom : atoms)
            {
                Xponge::ModelAtom atom;
                atom.name = String_Attr(py_atom, "name", "");
                atom.type = String_Attr(py_atom, "type", "");
                atom.x = Double_Attr(py_atom, "x");
                atom.y = Double_Attr(py_atom, "y");
                atom.z = Double_Attr(py_atom, "z");
                atom.charge = Double_Attr(py_atom, "charge");
                atom.mass = Double_Attr(py_atom, "mass");
                atom.lj_type = String_Attr(py_atom, "lj_type", atom.type);
                residue.atoms.push_back(atom);
                Py_DECREF(py_atom);
            }

            PyObject* bonds_object = Get_Attr(py_residue, "bonds");
            auto bonds = Iter_Items(bonds_object);
            Py_DECREF(bonds_object);
            for (PyObject* py_bond : bonds)
            {
                PyObject* first = PySequence_GetItem(py_bond, 0);
                PyObject* second = PySequence_GetItem(py_bond, 1);
                if (first == nullptr || second == nullptr)
                {
                    Py_XDECREF(first);
                    Py_XDECREF(second);
                    Py_DECREF(py_bond);
                    throw std::runtime_error("invalid bond tuple");
                }
                int i = static_cast<int>(PyLong_AsLong(first));
                int j = static_cast<int>(PyLong_AsLong(second));
                Py_DECREF(first);
                Py_DECREF(second);
                Py_DECREF(py_bond);
                if (PyErr_Occurred())
                {
                    throw std::runtime_error("invalid bond index");
                }
                if (i > j)
                {
                    std::swap(i, j);
                }
                residue.bonds.push_back({i, j});
            }
            molecule.residues.push_back(std::move(residue));
        }
        for (PyObject* py_residue : residues)
        {
            Py_DECREF(py_residue);
        }
    }
    catch (...)
    {
        for (PyObject* py_residue : residues)
        {
            Py_XDECREF(py_residue);
        }
        throw;
    }
    return molecule;
}

static PyObject* Mapping_Get_String(PyObject* mapping, const char* key)
{
    PyObject* key_object = PyUnicode_FromString(key);
    if (key_object == nullptr)
    {
        return nullptr;
    }
    PyObject* value = PyObject_GetItem(mapping, key_object);
    Py_DECREF(key_object);
    return value;
}

static double Dict_Double(PyObject* mapping, const char* key)
{
    PyObject* value = Mapping_Get_String(mapping, key);
    if (value == nullptr)
    {
        throw std::runtime_error(std::string("missing key ") + key);
    }
    const double result = PyFloat_AsDouble(value);
    Py_DECREF(value);
    if (PyErr_Occurred())
    {
        throw std::runtime_error(std::string("expected numeric key ") + key);
    }
    return result;
}

static bool Try_Dict_Double(PyObject* mapping, const char* key, double* result)
{
    PyObject* value = Mapping_Get_String(mapping, key);
    if (value == nullptr)
    {
        PyErr_Clear();
        return false;
    }
    *result = PyFloat_AsDouble(value);
    Py_DECREF(value);
    if (PyErr_Occurred())
    {
        PyErr_Clear();
        return false;
    }
    return true;
}

static int Dict_Int(PyObject* mapping, const char* key)
{
    PyObject* value = Mapping_Get_String(mapping, key);
    if (value == nullptr)
    {
        throw std::runtime_error(std::string("missing key ") + key);
    }
    const long result = PyLong_AsLong(value);
    Py_DECREF(value);
    if (PyErr_Occurred())
    {
        throw std::runtime_error(std::string("expected integer key ") + key);
    }
    return static_cast<int>(result);
}

static std::string Tuple_String(PyObject* tuple, Py_ssize_t index)
{
    PyObject* item = PySequence_GetItem(tuple, index);
    if (item == nullptr)
    {
        throw std::runtime_error("invalid tuple key");
    }
    std::string result = Object_To_String(item);
    Py_DECREF(item);
    return result;
}

static std::vector<PyObject*> Mapping_Items(PyObject* mapping)
{
    PyObject* items_object = PyMapping_Items(mapping);
    if (items_object == nullptr)
    {
        throw std::runtime_error("expected mapping");
    }
    auto items = Iter_Items(items_object);
    Py_DECREF(items_object);
    return items;
}

static Amber::GaffParameters Amber_Parameters_From_Python(
    PyObject* data_object,
    const Xponge::Molecule& molecule)
{
    Amber::GaffParameters parameters;
    for (const auto& atom : molecule.Atoms())
    {
        if (parameters.atom.find(atom.type) == parameters.atom.end())
        {
            parameters.atom[atom.type] = {atom.mass, atom.lj_type};
        }
    }

    PyObject* bond_table = Mapping_Get_String(data_object, "bond");
    PyObject* angle_table = Mapping_Get_String(data_object, "angle");
    PyObject* proper_table = Mapping_Get_String(data_object, "proper");
    PyObject* improper_table = Mapping_Get_String(data_object, "improper");
    PyObject* lj_table = Mapping_Get_String(data_object, "lj");
    if (!bond_table || !angle_table || !proper_table || !improper_table ||
        !lj_table)
    {
        Py_XDECREF(bond_table);
        Py_XDECREF(angle_table);
        Py_XDECREF(proper_table);
        Py_XDECREF(improper_table);
        Py_XDECREF(lj_table);
        throw std::runtime_error("invalid AMBER forcefield data");
    }
    try
    {
        for (PyObject* item : Mapping_Items(bond_table))
        {
            PyObject* key = PySequence_GetItem(item, 0);
            PyObject* value = PySequence_GetItem(item, 1);
            double k_value = 0.0;
            double b_value = 0.0;
            if (!Try_Dict_Double(value, "k", &k_value) ||
                !Try_Dict_Double(value, "b", &b_value) ||
                PySequence_Size(key) != 2)
            {
                Py_DECREF(key);
                Py_DECREF(value);
                Py_DECREF(item);
                continue;
            }
            parameters.bond[Amber::Canonical2(Tuple_String(key, 0),
                                              Tuple_String(key, 1))] = {
                k_value, b_value};
            Py_DECREF(key);
            Py_DECREF(value);
            Py_DECREF(item);
        }
        for (PyObject* item : Mapping_Items(angle_table))
        {
            PyObject* key = PySequence_GetItem(item, 0);
            PyObject* value = PySequence_GetItem(item, 1);
            double k_value = 0.0;
            double b_value = 0.0;
            if (!Try_Dict_Double(value, "k", &k_value) ||
                !Try_Dict_Double(value, "b", &b_value) ||
                PySequence_Size(key) != 3)
            {
                Py_DECREF(key);
                Py_DECREF(value);
                Py_DECREF(item);
                continue;
            }
            parameters.angle[Amber::Canonical3(
                Tuple_String(key, 0), Tuple_String(key, 1),
                Tuple_String(key, 2))] = {k_value, b_value};
            Py_DECREF(key);
            Py_DECREF(value);
            Py_DECREF(item);
        }
        for (PyObject* item : Mapping_Items(proper_table))
        {
            PyObject* key = PySequence_GetItem(item, 0);
            PyObject* value = PySequence_GetItem(item, 1);
            if (PySequence_Size(key) != 4)
            {
                Py_DECREF(key);
                Py_DECREF(value);
                Py_DECREF(item);
                continue;
            }
            PyObject* ks = Mapping_Get_String(value, "ks");
            PyObject* phi0s = Mapping_Get_String(value, "phi0s");
            PyObject* periodicitys = Mapping_Get_String(value, "periodicitys");
            auto ks_items = Iter_Items(ks);
            auto phi_items = Iter_Items(phi0s);
            auto per_items = Iter_Items(periodicitys);
            std::vector<Amber::ProperTerm> terms;
            for (std::size_t i = 0; i < ks_items.size(); ++i)
            {
                terms.push_back({PyFloat_AsDouble(ks_items[i]),
                                 PyFloat_AsDouble(phi_items[i]),
                                 static_cast<int>(
                                     PyLong_AsLong(per_items[i]))});
            }
            for (auto* x : ks_items) Py_DECREF(x);
            for (auto* x : phi_items) Py_DECREF(x);
            for (auto* x : per_items) Py_DECREF(x);
            Py_DECREF(ks);
            Py_DECREF(phi0s);
            Py_DECREF(periodicitys);
            parameters.proper[std::make_tuple(
                Tuple_String(key, 0), Tuple_String(key, 1),
                Tuple_String(key, 2), Tuple_String(key, 3))] = terms;
            Py_DECREF(key);
            Py_DECREF(value);
            Py_DECREF(item);
        }
        for (PyObject* item : Mapping_Items(improper_table))
        {
            PyObject* key = PySequence_GetItem(item, 0);
            PyObject* value = PySequence_GetItem(item, 1);
            double k_value = 0.0;
            double phi_value = 0.0;
            if (!Try_Dict_Double(value, "k", &k_value) ||
                !Try_Dict_Double(value, "phi0", &phi_value) ||
                PySequence_Size(key) != 4)
            {
                Py_DECREF(key);
                Py_DECREF(value);
                Py_DECREF(item);
                continue;
            }
            parameters.improper[std::make_tuple(
                Tuple_String(key, 0), Tuple_String(key, 1),
                Tuple_String(key, 2), Tuple_String(key, 3))] = {
                k_value, phi_value, Dict_Int(value, "periodicity")};
            Py_DECREF(key);
            Py_DECREF(value);
            Py_DECREF(item);
        }
        for (PyObject* item : Mapping_Items(lj_table))
        {
            PyObject* key = PySequence_GetItem(item, 0);
            PyObject* value = PySequence_GetItem(item, 1);
            std::string name = Object_To_String(key);
            const auto dash = name.find('-');
            if (dash != std::string::npos)
            {
                name = name.substr(0, dash);
            }
            parameters.lj[name] = {Dict_Double(value, "epsilon"),
                                   Dict_Double(value, "rmin")};
            Py_DECREF(key);
            Py_DECREF(value);
            Py_DECREF(item);
        }
    }
    catch (...)
    {
        Py_DECREF(bond_table);
        Py_DECREF(angle_table);
        Py_DECREF(proper_table);
        Py_DECREF(improper_table);
        Py_DECREF(lj_table);
        throw;
    }
    Py_DECREF(bond_table);
    Py_DECREF(angle_table);
    Py_DECREF(proper_table);
    Py_DECREF(improper_table);
    Py_DECREF(lj_table);
    return parameters;
}

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

static PyObject* Module_generate_gaff_frcmod(PyObject*,
                                             PyObject* args,
                                             PyObject* kwargs)
{
    const char* ifname = nullptr;
    const char* ofname = nullptr;
    int ffset = 1;
    int print_all = 0;
    int print_dihedral_contain_X = 1;
    PyObject* datapath = Py_None;
    static const char* keywords[] = {"ifname",
                                     "ofname",
                                     "ffset",
                                     "print_all",
                                     "print_dihedral_contain_X",
                                     "datapath",
                                     nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "ss|ippO", const_cast<char**>(keywords), &ifname,
            &ofname, &ffset, &print_all, &print_dihedral_contain_X, &datapath))
    {
        return nullptr;
    }
    return Guarded([&]() -> PyObject* {
        Amber::Parmchk2Options options;
        options.ffset = ffset;
        options.print_all = print_all != 0;
        options.print_dihedral_contain_X = print_dihedral_contain_X != 0;
        if (datapath != Py_None)
        {
            PyObject* utf8 = PyUnicode_AsUTF8String(datapath);
            if (utf8 == nullptr)
            {
                return nullptr;
            }
            const char* path = PyBytes_AsString(utf8);
            if (path == nullptr)
            {
                Py_DECREF(utf8);
                return nullptr;
            }
            options.datapath = path;
            Py_DECREF(utf8);
        }
        Amber::Generate_Gaff_Frcmod(ifname, ofname, options);
        Py_RETURN_NONE;
    });
}

static PyObject* Module_load_gaff_parameters(PyObject*,
                                             PyObject* args,
                                             PyObject* kwargs)
{
    const char* dat_path = nullptr;
    PyObject* frcmod_path = Py_None;
    static const char* keywords[] = {"dat_path", "frcmod_path", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|O",
                                     const_cast<char**>(keywords), &dat_path,
                                     &frcmod_path))
    {
        return nullptr;
    }
    return Guarded([&] {
        std::string frcmod;
        if (frcmod_path != Py_None)
        {
            frcmod = Object_To_String(frcmod_path);
        }
        return New_PyGaffParameters(
            Amber::Load_Gaff_Parameters(dat_path, frcmod));
    });
}

static PyObject* Module_load_frcmod(PyObject*, PyObject* args)
{
    const char* filename = nullptr;
    if (!PyArg_ParseTuple(args, "s", &filename))
    {
        return nullptr;
    }
    return Guarded([&] {
        const auto data = Amber::Load_Frcmod_As_Xponge_Data(filename);
        PyObject* result = PyList_New(7);
        if (result == nullptr)
        {
            return static_cast<PyObject*>(nullptr);
        }
        for (std::size_t i = 0; i < data.sections.size(); ++i)
        {
            PyObject* value = PyUnicode_FromString(data.sections[i].c_str());
            if (value == nullptr)
            {
                Py_DECREF(result);
                return static_cast<PyObject*>(nullptr);
            }
            PyList_SetItem(result, static_cast<Py_ssize_t>(i), value);
        }
        PyObject* cmap = PyDict_New();
        if (cmap == nullptr)
        {
            Py_DECREF(result);
            return static_cast<PyObject*>(nullptr);
        }
        for (const auto& item : data.cmap)
        {
            PyObject* entry = PyDict_New();
            PyObject* resolution = PyLong_FromLong(item.second.resolution);
            PyObject* parameters =
                PyList_New(static_cast<Py_ssize_t>(
                    item.second.parameters.size()));
            if (entry == nullptr || resolution == nullptr ||
                parameters == nullptr)
            {
                Py_XDECREF(entry);
                Py_XDECREF(resolution);
                Py_XDECREF(parameters);
                Py_DECREF(cmap);
                Py_DECREF(result);
                return static_cast<PyObject*>(nullptr);
            }
            for (std::size_t i = 0; i < item.second.parameters.size(); ++i)
            {
                PyObject* value = PyFloat_FromDouble(
                    item.second.parameters[i]);
                if (value == nullptr)
                {
                    Py_DECREF(entry);
                    Py_DECREF(resolution);
                    Py_DECREF(parameters);
                    Py_DECREF(cmap);
                    Py_DECREF(result);
                    return static_cast<PyObject*>(nullptr);
                }
                PyList_SetItem(parameters, static_cast<Py_ssize_t>(i), value);
            }
            if (PyDict_SetItemString(entry, "resolution", resolution) < 0 ||
                PyDict_SetItemString(entry, "parameters", parameters) < 0)
            {
                Py_DECREF(entry);
                Py_DECREF(resolution);
                Py_DECREF(parameters);
                Py_DECREF(cmap);
                Py_DECREF(result);
                return static_cast<PyObject*>(nullptr);
            }
            Py_DECREF(resolution);
            Py_DECREF(parameters);
            if (PyDict_SetItemString(cmap, item.first.c_str(), entry) < 0)
            {
                Py_DECREF(entry);
                Py_DECREF(cmap);
                Py_DECREF(result);
                return static_cast<PyObject*>(nullptr);
            }
            Py_DECREF(entry);
        }
        PyList_SetItem(result, 6, cmap);
        return result;
    });
}

static PyObject* Module_save_sponge_input(PyObject*,
                                          PyObject* args,
                                          PyObject* kwargs)
{
    PyObject* molecule_object = nullptr;
    const char* output_dir = ".";
    PyObject* parameters_object = Py_None;
    const char* prefix = "xponge";
    PyObject* box_object = Py_None;
    static const char* keywords[] = {
        "molecule", "output_dir", "parameters", "prefix", "box", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|sOsO",
                                     const_cast<char**>(keywords),
                                     &molecule_object, &output_dir,
                                     &parameters_object, &prefix, &box_object))
    {
        return nullptr;
    }
    if (parameters_object == Py_None)
    {
        PyErr_SetString(PyExc_NotImplementedError,
                        "native save_sponge_input currently requires "
                        "explicit parameters");
        return nullptr;
    }
    if (Py_TYPE(parameters_object) !=
        reinterpret_cast<PyTypeObject*>(PyGaffParametersType))
    {
        PyErr_SetString(PyExc_TypeError,
                        "parameters must be xponge2.GaffParameters");
        return nullptr;
    }
    return Guarded([&]() -> PyObject* {
        Xponge::Molecule molecule = Molecule_From_Python(molecule_object);
        Amber::SpongeInputOptions options;
        options.output_dir = output_dir;
        options.prefix = prefix;
        if (box_object != Py_None)
        {
            auto items = Iter_Items(box_object);
            if (items.size() != 6)
            {
                for (auto* item : items) Py_DECREF(item);
                throw std::runtime_error("box must contain six values");
            }
            options.box.clear();
            for (auto* item : items)
            {
                options.box.push_back(PyFloat_AsDouble(item));
                Py_DECREF(item);
                if (PyErr_Occurred())
                {
                    throw std::runtime_error("box must contain floats");
                }
            }
        }
        Amber::Save_Gaff_Sponge_Input(
            molecule,
            Require_Gaff_Parameters(
                reinterpret_cast<PyGaffParametersObject*>(parameters_object)),
            options);
        Py_RETURN_NONE;
    });
}

static PyObject* Module_save_amber_sponge_input(PyObject*,
                                                PyObject* args,
                                                PyObject* kwargs)
{
    PyObject* molecule_object = nullptr;
    PyObject* data_object = nullptr;
    const char* output_dir = ".";
    const char* prefix = "xponge";
    PyObject* box_object = Py_None;
    const char* cmap_source = "";
    static const char* keywords[] = {"molecule", "data", "output_dir", "prefix",
                                     "box", "cmap_source", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|ssOs",
                                     const_cast<char**>(keywords),
                                     &molecule_object, &data_object,
                                     &output_dir, &prefix, &box_object,
                                     &cmap_source))
    {
        return nullptr;
    }
    return Guarded([&]() -> PyObject* {
        Xponge::Molecule molecule = Molecule_From_Python(molecule_object);
        Amber::GaffParameters parameters =
            Amber_Parameters_From_Python(data_object, molecule);
        Amber::SpongeInputOptions options;
        options.output_dir = output_dir;
        options.prefix = prefix;
        options.connect_residue_tails = true;
        options.prefix_files = true;
        options.write_mdin = false;
        options.write_atom_metadata = true;
        options.charge_scale = 18.2223;
        options.cmap_source = cmap_source;
        if (box_object != Py_None)
        {
            auto items = Iter_Items(box_object);
            if (items.size() != 6)
            {
                for (auto* item : items) Py_DECREF(item);
                throw std::runtime_error("box must contain six values");
            }
            options.box.clear();
            for (auto* item : items)
            {
                options.box.push_back(PyFloat_AsDouble(item));
                Py_DECREF(item);
                if (PyErr_Occurred())
                {
                    throw std::runtime_error("box must contain floats");
                }
            }
        }
        Amber::Save_Gaff_Sponge_Input(molecule, parameters, options);
        Py_RETURN_NONE;
    });
}

static PyObject* Module_assignment_to_residue_type(PyObject*,
                                                   PyObject* args,
                                                   PyObject* kwargs)
{
    PyObject* assign_object = nullptr;
    PyObject* name_object = Py_None;
    PyObject* charge_object = Py_None;
    static const char* keywords[] = {"assign", "name", "charge", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO",
                                     const_cast<char**>(keywords),
                                     &assign_object, &name_object,
                                     &charge_object))
    {
        return nullptr;
    }
    if (Py_TYPE(assign_object) != reinterpret_cast<PyTypeObject*>(PyAssignType))
    {
        PyErr_SetString(PyExc_TypeError, "assign must be xponge2.Assign");
        return nullptr;
    }
    return Guarded([&]() -> PyObject* {
        const XA::Assignment& assignment = Require_Assignment(
            reinterpret_cast<PyAssignObject*>(assign_object));
        std::string residue_name = assignment.name();
        if (name_object != Py_None)
        {
            residue_name = Object_To_String(name_object);
        }
        PyObject* core_module = PyImport_ImportModule("xponge2.core");
        if (core_module == nullptr)
        {
            return nullptr;
        }
        PyObject* residue_type_class =
            PyObject_GetAttrString(core_module, "ResidueType");
        Py_DECREF(core_module);
        if (residue_type_class == nullptr)
        {
            return nullptr;
        }
        PyObject* residue_name_object =
            PyUnicode_FromString(residue_name.c_str());
        if (residue_name_object == nullptr)
        {
            Py_DECREF(residue_type_class);
            return nullptr;
        }
        PyObject* residue =
            PyObject_CallFunctionObjArgs(residue_type_class,
                                         residue_name_object, nullptr);
        Py_DECREF(residue_name_object);
        Py_DECREF(residue_type_class);
        if (residue == nullptr)
        {
            return nullptr;
        }

        const auto& atoms = assignment.atoms();
        const auto& atom_types = assignment.atom_types();
        for (std::size_t i = 0; i < assignment.atom_numbers(); ++i)
        {
            double charge = atoms[i].charge;
            if (charge_object != Py_None)
            {
                PyObject* item =
                    PySequence_GetItem(charge_object, static_cast<Py_ssize_t>(i));
                if (item == nullptr)
                {
                    Py_DECREF(residue);
                    return nullptr;
                }
                charge = PyFloat_AsDouble(item);
                Py_DECREF(item);
                if (PyErr_Occurred())
                {
                    Py_DECREF(residue);
                    return nullptr;
                }
            }
            const std::string atom_name = std::to_string(i + 1);
            const auto& coord = atoms[i].coordinate;
            PyObject* added = PyObject_CallMethod(
                residue, "add_atom", "ssdddd", atom_name.c_str(),
                atom_types[i].c_str(), coord.x, coord.y, coord.z, charge);
            if (added == nullptr)
            {
                Py_DECREF(residue);
                return nullptr;
            }
            Py_DECREF(added);
        }

        const auto& bonds = assignment.bonds();
        for (std::size_t i = 0; i < bonds.size(); ++i)
        {
            for (const auto& item : bonds[i])
            {
                if (static_cast<int>(i) >= item.first)
                {
                    continue;
                }
                PyObject* added = PyObject_CallMethod(
                    residue, "add_connectivity", "ii", static_cast<int>(i),
                    item.first);
                if (added == nullptr)
                {
                    Py_DECREF(residue);
                    return nullptr;
                }
                Py_DECREF(added);
            }
        }
        return residue;
    });
}

PyDoc_STRVAR(Module_get_assignment_from_mol2_doc,
             "get_assignment_from_mol2(file, total_charge=None)\n"
             "--\n\nRead a mol2 file as an Assign object.");
PyDoc_STRVAR(Module_generate_gaff_frcmod_doc,
             "generate_gaff_frcmod(ifname, ofname, ffset=1, "
             "print_all=False, print_dihedral_contain_X=True, "
             "datapath=None)\n"
             "--\n\nGenerate a GAFF frcmod file from a GAFF-typed mol2 file.");
PyDoc_STRVAR(Module_load_gaff_parameters_doc,
             "load_gaff_parameters(dat_path, frcmod_path=None)\n"
             "--\n\nLoad GAFF parameters into a native parameter table.");
PyDoc_STRVAR(Module_load_frcmod_doc,
             "load_frcmod(filename)\n"
             "--\n\nLoad an frcmod file in Xponge-compatible string form.");
PyDoc_STRVAR(Module_save_sponge_input_doc,
             "save_sponge_input(molecule, output_dir='.', parameters=None, "
             "prefix='xponge', box=None)\n"
             "--\n\nWrite SPONGE input files from a molecule.");
PyDoc_STRVAR(Module_assignment_to_residue_type_doc,
             "assignment_to_residue_type(assign, name=None, charge=None)\n"
             "--\n\nConvert an Assign object to a ResidueType.");
PyDoc_STRVAR(Module_save_amber_sponge_input_doc,
             "save_amber_sponge_input(molecule, data, output_dir='.', "
             "prefix='xponge', box=None, cmap_source='')\n"
             "--\n\nWrite AMBER biopolymer SPONGE input files.");

static PyMethodDef Xponge2Methods[] = {
    {"get_assignment_from_mol2",
     reinterpret_cast<PyCFunction>(Module_get_assignment_from_mol2),
     METH_VARARGS | METH_KEYWORDS, Module_get_assignment_from_mol2_doc},
    {"Get_Assignment_From_Mol2",
     reinterpret_cast<PyCFunction>(Module_get_assignment_from_mol2),
     METH_VARARGS | METH_KEYWORDS, Module_get_assignment_from_mol2_doc},
    {"generate_gaff_frcmod",
     reinterpret_cast<PyCFunction>(Module_generate_gaff_frcmod),
     METH_VARARGS | METH_KEYWORDS, Module_generate_gaff_frcmod_doc},
    {"load_gaff_parameters",
     reinterpret_cast<PyCFunction>(Module_load_gaff_parameters),
     METH_VARARGS | METH_KEYWORDS, Module_load_gaff_parameters_doc},
    {"load_frcmod", reinterpret_cast<PyCFunction>(Module_load_frcmod),
     METH_VARARGS, Module_load_frcmod_doc},
    {"save_sponge_input", reinterpret_cast<PyCFunction>(Module_save_sponge_input),
     METH_VARARGS | METH_KEYWORDS, Module_save_sponge_input_doc},
    {"assignment_to_residue_type",
     reinterpret_cast<PyCFunction>(Module_assignment_to_residue_type),
     METH_VARARGS | METH_KEYWORDS, Module_assignment_to_residue_type_doc},
    {"save_amber_sponge_input",
     reinterpret_cast<PyCFunction>(Module_save_amber_sponge_input),
     METH_VARARGS | METH_KEYWORDS, Module_save_amber_sponge_input_doc},
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
    PyGaffParametersType = PyType_FromSpec(&PyGaffParametersTypeSpec);
    if (PyGaffParametersType == nullptr)
    {
        Py_DECREF(PyAssignType);
        Py_DECREF(module);
        return nullptr;
    }
    if (PyModule_AddObject(module, "GaffParameters",
                           Py_NewRef(PyGaffParametersType)) < 0)
    {
        Py_DECREF(PyGaffParametersType);
        Py_DECREF(PyAssignType);
        Py_DECREF(module);
        return nullptr;
    }
    PyModule_AddStringConstant(module, "__version__", XPONGE2_VERSION);
    return module;
}
