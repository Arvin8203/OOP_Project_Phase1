
// main.cpp
// ------------
// This program reads a small “menu” from InputFile.txt, expects commands such as
//   add R1 1 2 1000
//   add D1 2 3
//   solve dc
//   tran 0.0001 0 0.01
//   delete R3
//   list
//   .end
//
// It builds a circuit in memory using OOP classes.  DC solves use Modified Nodal Analysis
// with a continuous‐diode exponential model inside Newton‐Raphson.  Transient (BACKWARD_EULER)
// is also provided (diodes are “frozen” at their DC operating point each time step).
//
// Usage:
//   – Edit the two paths near the top of main(): inputPath and outputPath.
//   – Build with your favorite C++17/20 toolchain.
//   – Run; the program writes all text to OutputFile.txt.
//-----------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <dirent.h>
#include <map>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

//-----------------------------------------------------------------------------
//   Global: where to read commands and write output
//-----------------------------------------------------------------------------
static const string schematicsDir = "schematics/";
static const string inputPath = schematicsDir + "input01.txt";
static const string outputPath = schematicsDir + "output01.txt";

//-----------------------------------------------------------------------------
//   Enumeration of element types & integration methods
//-----------------------------------------------------------------------------
enum ElementType {
    RESISTOR,
    CAPACITOR,
    INDUCTOR,
    DIODE,
    VSOURCE,
    ISOURCE,
    GROUND,
    DEP_VCVS,  // E: voltage‐controlled voltage source
    DEP_VCCS,  // G: voltage‐controlled current source
    DEP_CCVS,  // H: current‐controlled voltage source
    DEP_CCCS   // F: current‐controlled current source
};

enum IntegrationMethod { BACKWARD_EULER, TRAPEZOIDAL };

//-----------------------------------------------------------------------------
//   Base class for exception handling all circuit elements
//-----------------------------------------------------------------------------
class NameException : public exception {
    string msg;
public:
    NameException(const string& m) : msg(m) {}
    NameException() : msg("Error: Element not found in library") {}

    const char* what() const noexcept override {
            return msg.c_str();
    }
};


class ValueException : public exception {
private:
    string elementType;
public:
    ValueException(const string& type) : elementType(type) {}
    const char* what() const noexcept override {
            if (elementType == "resistor") {
                return "Error: Resistance cannot be zero or negative";
            } else if (elementType == "capacitor") {
                return "Error: Capacitance cannot be zero or negative";
            } else if (elementType == "inductor") {
                return "Error: Inductance cannot be zero or negative";
            }
            return "Error: Invalid value";
    }
};

class SyntaxException : public exception {
public:
    const char* what() const noexcept override {
            return "Error: Syntax error";
    }
};

class ModelException : public exception {
public:
    const char* what() const noexcept override {
            return "Error: Model not found in library";
    }
};

class DuplicateException : public exception {
private:
    string elementType;
    string name;
public:
    DuplicateException(const string& type, const string& nm) : elementType(type), name(nm) {}
    const char* what() const noexcept override {
            static string msg;
            msg = "Error: " + elementType + " " + name + " already exists in the circuit";
            return msg.c_str();
    }
};

class NotFoundException : public exception {
private:
    string elementType;
public:
    NotFoundException(const string& type) : elementType(type) {}
    const char* what() const noexcept override {
            static string msg;
            msg = "Error: Cannot delete " + elementType + "; component not found";
            return msg.c_str();
    }
};

class CircuitElement {
public:
    string       name;
    ElementType  type;
    int          node1, node2;        // for a two‐terminal element; for GROUND, node2==0

    CircuitElement(const string& n, ElementType t, int n1, int n2)
            : name(n), type(t), node1(n1), node2(n2) {}

    virtual ~CircuitElement() = default;
};

//-----------------------------------------------------------------------------
//   Resistor
//-----------------------------------------------------------------------------
class Resistor : public CircuitElement {
public:
    double resistance;
    Resistor(const string& n, int n1, int n2, double r)
            : CircuitElement(n, RESISTOR, n1, n2), resistance(r) {}
};

//-----------------------------------------------------------------------------
//   Capacitor
//-----------------------------------------------------------------------------
class Capacitor : public CircuitElement {
public:
    double capacitance;
    double lastVoltageDiff;   // V(n1)–V(n2) at previous time step
    double lastCurrent;       // i_C at previous time step
    Capacitor(const string& n, int n1, int n2, double c)
            : CircuitElement(n, CAPACITOR, n1, n2),
              capacitance(c), lastVoltageDiff(0.0), lastCurrent(0.0) {}
};

//-----------------------------------------------------------------------------
//   Inductor
//-----------------------------------------------------------------------------
class Inductor : public CircuitElement {
public:
    double inductance;
    double lastCurrent;       // i_L at previous time step
    double lastVoltage;       // V(n1)–V(n2) at previous time step
    Inductor(const string& n, int n1, int n2, double L)
            : CircuitElement(n, INDUCTOR, n1, n2),
              inductance(L), lastCurrent(0.0), lastVoltage(0.0) {}
};

//-----------------------------------------------------------------------------
//   Diode (PN junction, exponential model)
//     I = Is ( e^{ Vd / (n·Vt) } – 1 )
//     dI/dV = Gd = (Is/(n·Vt)) e^{ Vd / (n·Vt) }
//-----------------------------------------------------------------------------
class Diode : public CircuitElement {
public:
    double saturationCurrent;    // Is
    double thermalVoltage;       // Vt
    double emissionCoeff;        // n
    Diode(const string& n, int a, int c,
          double Is_ = 1e-14, double ncoef_ = 1.0, double Vt_ = 0.02585)
            : CircuitElement(n, DIODE, a, c),
              saturationCurrent(Is_), thermalVoltage(Vt_), emissionCoeff(ncoef_) {}
};

//-----------------------------------------------------------------------------
//   Independent voltage source (ideal, dc value “voltage”)
//-----------------------------------------------------------------------------
class VoltageSource : public CircuitElement {
public:
    double voltage;
    VoltageSource(const string& n, int p, int q, double v)
            : CircuitElement(n, VSOURCE, p, q), voltage(v) {}
};

//-----------------------------------------------------------------------------
//   Independent current source (ideal, dc value “current” from n1 → n2)
//-----------------------------------------------------------------------------
class CurrentSource : public CircuitElement {
public:
    double current;
    CurrentSource(const string& n, int p, int q, double i)
            : CircuitElement(n, ISOURCE, p, q), current(i) {}
};

//-----------------------------------------------------------------------------
//   Dependent sources
//-----------------------------------------------------------------------------

// E: voltage‐controlled voltage source
class VCVS : public CircuitElement {
public:
    int ctrlNode1, ctrlNode2;
    double gain;
    VCVS(const string& n, int n1, int n2,
         int cn1, int cn2, double g)
            : CircuitElement(n, DEP_VCVS, n1, n2),
              ctrlNode1(cn1), ctrlNode2(cn2), gain(g) {}
};

// G: voltage‐controlled current source
class VCCS : public CircuitElement {
public:
    int ctrlNode1, ctrlNode2;
    double gain;
    VCCS(const string& n, int n1, int n2,
         int cn1, int cn2, double g)
            : CircuitElement(n, DEP_VCCS, n1, n2),
              ctrlNode1(cn1), ctrlNode2(cn2), gain(g) {}
};

// H: current‐controlled voltage source
class CCVS : public CircuitElement {
public:
    string vName;
    double gain;
    CCVS(const string& n, int n1, int n2,
         const string& ctrl, double g)
            : CircuitElement(n, DEP_CCVS, n1, n2),
              vName(ctrl), gain(g) {}
};

// F: current‐controlled current source
class CCCS : public CircuitElement {
public:
    string vName;
    double gain;
    CCCS(const string& n, int n1, int n2,
         const string& ctrl, double g)
            : CircuitElement(n, DEP_CCCS, n1, n2),
              vName(ctrl), gain(g) {}
};

//-----------------------------------------------------------------------------
//   Ground (zero‐volt reference at a single node)
//     We treat ground as a direct fixed‐voltage constraint in MNA.
//-----------------------------------------------------------------------------
class Ground : public CircuitElement {
public:
    Ground(const string& n, int nd) : CircuitElement(n, GROUND, nd, 0) {}
};

//-----------------------------------------------------------------------------
//   Simple Gaussian‐Elimination solver for Ax=b  (returns false if singular)
//-----------------------------------------------------------------------------
class GaussianSolver {
public:
    // Solve A x = b in place.  Returns false if matrix is singular.
    static bool solve(vector<vector<double>>& A, vector<double>& b, vector<double>& x) {
        int n = (int)A.size();
        x.assign(n, 0.0);

        // Forward elimination with partial pivoting
        for (int k = 0; k < n; ++k) {
            // find pivot row
            double maxVal = fabs(A[k][k]);
            int pivot = k;
            for (int i = k + 1; i < n; ++i) {
                double val = fabs(A[i][k]);
                if (val > maxVal) {
                    maxVal = val;
                    pivot = i;
                }
            }
            if (maxVal < 1e-14) {
                return false;
            }
            // swap if needed
            if (pivot != k) {
                swap(A[k], A[pivot]);
                swap(b[k], b[pivot]);
            }
            // eliminate below
            for (int i = k + 1; i < n; ++i) {
                double factor = A[i][k] / A[k][k];
                b[i] -= factor * b[k];
                for (int j = k; j < n; ++j) {
                    A[i][j] -= factor * A[k][j];
                }
            }
        }

        // Back‐substitution
        for (int i = n - 1; i >= 0; --i) {
            double sum = 0.0;
            for (int j = i + 1; j < n; ++j) {
                sum += A[i][j] * x[j];
            }
            x[i] = (b[i] - sum) / A[i][i];
        }
        return true;
    }
};

//-----------------------------------------------------------------------------
//   The main Circuit class: holds all elements, builds MNA matrices, does DC and transient solves.
//-----------------------------------------------------------------------------
class Circuit {
private:
    vector<CircuitElement*> elements;
    map<int,int>           nodeIndex;     // maps actual node number → index [0..N−1], except ground(0)
    vector<int>            indexToNode;   // reverse map: index → actual node number
    int                    nodeCount;     // how many non‐zero nodes

public:
    Circuit() : nodeCount(0) {}
    ~Circuit() {
        for (auto e : elements) delete e;
    }

    // Given an integer “node”.  We reserve index “−1” for ground(0).
    // If node != 0 and not yet assigned, assign a new index.
    int getNodeIndex(int node) {
        if (node == 0) return -1;
        auto it = nodeIndex.find(node);
        if (it == nodeIndex.end()) {
            int idx = nodeCount;
            nodeIndex[node] = idx;
            indexToNode.push_back(node);
            nodeCount++;
            return idx;
        }
        return it->second;
    }

    // Add various element types; each call ensures both endpoints (even if ground) get their index.
    void addResistor(const string& name, int n1, int n2, double r) {
        if (hasElement(name)) {
            throw DuplicateException("resistor", name);
        }
        if (r <= 0) {
            throw ValueException("resistor");
        }
        getNodeIndex(n1);
        getNodeIndex(n2);
        elements.push_back(new Resistor(name, n1, n2, r));
    }

    void addCapacitor(const string& name, int n1, int n2, double c) {
        if (hasElement(name)) {
            throw DuplicateException("capacitor", name);
        }
        if (c <= 0) {
            throw ValueException("capacitor");
        }
        getNodeIndex(n1);
        getNodeIndex(n2);
        elements.push_back(new Capacitor(name, n1, n2, c));
    }

    void addInductor(const string& name, int n1, int n2, double L) {
        if (hasElement(name)) {
            throw DuplicateException("inductor", name);
        }
        if (L <= 0) {
            throw ValueException("inductor");
        }
        getNodeIndex(n1);
        getNodeIndex(n2);
        elements.push_back(new Inductor(name, n1, n2, L));
    }

    void addDiode(const string& name, int a, int c, double Is = 1e-14, double ncoef = 1.0, double Vt = 0.02585) {
        if (hasElement(name)) {
            throw DuplicateException("diode", name);
        }
        getNodeIndex(a);
        getNodeIndex(c);
        elements.push_back(new Diode(name, a, c, Is, ncoef, Vt));
    }

    void addVoltageSource(const string& name, int p, int q, double v) {
        if (hasElement(name)) {
            throw DuplicateException("voltage source", name);
        }
        getNodeIndex(p);
        getNodeIndex(q);
        elements.push_back(new VoltageSource(name, p, q, v));
    }

    void addCurrentSource(const string& name, int p, int q, double i) {
        if (hasElement(name)) {
            throw DuplicateException("current source", name);
        }
        getNodeIndex(p);
        getNodeIndex(q);
        elements.push_back(new CurrentSource(name, p, q, i));
    }

    void addGround(const string& name, int nd) {
        if (hasElement(name)) {
            throw DuplicateException("ground", name);
        }
        getNodeIndex(nd);
        elements.push_back(new Ground(name, nd));
    }

    // E: voltage‐controlled voltage source
    void addVCVS(const string& name, int n1, int n2, int cn1, int cn2, double g){
        if (nodeIndex.count(cn1) == 0 || nodeIndex.count(cn2) == 0) {
            throw NameException("Error: Undefined control node for VCVS");
        }
        getNodeIndex(n1);
        getNodeIndex(n2);
        getNodeIndex(cn1);
        getNodeIndex(cn2);
        elements.push_back(new VCVS(name, n1, n2, cn1, cn2, g));
    }

    // G: voltage‐controlled current source
    void addVCCS(const string& name, int n1, int n2, int cn1, int cn2, double g){
        if (nodeIndex.count(cn1) == 0 || nodeIndex.count(cn2) == 0) {
            throw NameException("Error: Undefined control node for VCCS");
        }
        getNodeIndex(n1);
        getNodeIndex(n2);
        getNodeIndex(cn1);
        getNodeIndex(cn2);
        elements.push_back(new VCCS(name, n1, n2, cn1, cn2, g));
    }

    // H: current‐controlled voltage source
    void addCCVS(const string& name, int n1, int n2, const string& vName, double g){
        if (!findElement(vName)) {
            throw NameException("Error: Controlling source '" + vName + "' not found");
        }
        getNodeIndex(n1);
        getNodeIndex(n2);
        elements.push_back(new CCVS(name, n1, n2, vName, g));
    }

    // F: current‐controlled current source
    void addCCCS(const string& name, int n1, int n2, const string& vName, double g){
        if (!findElement(vName)) {
            throw NameException("Error: Controlling source '" + vName + "' not found");
        }
        getNodeIndex(n1);
        getNodeIndex(n2);
        elements.push_back(new CCCS(name, n1, n2, vName, g));
    }

    // Find by name
    CircuitElement* findElement(const string& nm) {
        for (auto e : elements)
            if (e->name == nm) return e;
        return nullptr;
    }
    // Delete by name
    bool deleteElement(const string& nm) {
        for (size_t i = 0; i < elements.size(); ++i) {
            if (elements[i]->name == nm) {
                delete elements[i];
                elements.erase(elements.begin() + i);
                return true;
            }
        }
        string typeStr;
        switch (nm[0]) {
            case 'R': typeStr = "resistor"; break;
            case 'C': typeStr = "capacitor"; break;
            case 'L': typeStr = "inductor"; break;
            case 'D': typeStr = "diode"; break;
            case 'V': typeStr = "voltage source"; break;
            case 'I': typeStr = "current source"; break;
            case 'G': typeStr = "ground"; break;
            default: typeStr = "component"; break;
        }
        throw NotFoundException(typeStr);
    }

    bool hasElement(const string& nm) const {
        for (auto e : elements)
            if (e->name == nm) return true;
        return false;
    }

    // List everything
    void listElements(ofstream& out) const {
        if (elements.empty()) {
            out << "No elements in the circuit.\n";
            return;
        }
        out << "Circuit elements:\n";
        for (auto e : elements) {
            switch (e->type) {
                case RESISTOR: {
                    auto r = static_cast<Resistor*>(e);
                    out << "  " << r->name << ": Resistor "
                        << r->node1 << "-" << r->node2
                        << ", " << r->resistance << " Ohm\n";
                    break;
                }
                case CAPACITOR: {
                    auto c = static_cast<Capacitor*>(e);
                    out << "  " << c->name << ": Capacitor "
                        << c->node1 << "-" << c->node2
                        << ", " << c->capacitance << " F\n";
                    break;
                }
                case INDUCTOR: {
                    auto l = static_cast<Inductor*>(e);
                    out << "  " << l->name << ": Inductor "
                        << l->node1 << "-" << l->node2
                        << ", " << l->inductance << " H\n";
                    break;
                }
                case DIODE: {
                    auto d = static_cast<Diode*>(e);
                    out << "  " << d->name << ": Diode "
                        << d->node1 << "->" << d->node2
                        << ", Is=" << d->saturationCurrent
                        << ", n=" << d->emissionCoeff
                        << ", Vt=" << d->thermalVoltage << " V\n";
                    break;
                }
                case VSOURCE: {
                    auto v = static_cast<VoltageSource*>(e);
                    out << "  " << v->name << ": Vsrc "
                        << v->node1 << "-" << v->node2
                        << " = " << v->voltage << " V\n";
                    break;
                }
                case ISOURCE: {
                    auto i = static_cast<CurrentSource*>(e);
                    out << "  " << i->name << ": Isrc "
                        << i->node1 << "->" << i->node2
                        << " = " << i->current << " A\n";
                    break;
                }
                case GROUND: {
                    auto g = static_cast<Ground*>(e);
                    out << "  " << g->name << ": Ground at node "
                        << g->node1 << "\n";
                    break;
                }
                    // VCVS (E)
                case DEP_VCVS: {
                    auto d = static_cast<VCVS*>(e);
                    out << "  " << d->name << ": VCVS " << d->node1 << "-" << d->node2 << ", ctrl nodes " << d->ctrlNode1 << "-" << d->ctrlNode2 << ", gain=" << d->gain << "\n";
                    break;
                }
                    // VCCS (G)
                case DEP_VCCS: {
                    auto d = static_cast<VCCS*>(e);
                    out << "  " << d->name << ": VCCS " << d->node1 << "-" << d->node2 << ", ctrl nodes " << d->ctrlNode1 << "-" << d->ctrlNode2 << ", gain=" << d->gain << "\n";
                    break;
                }
                    // CCVS (H)
                case DEP_CCVS: {
                    auto d = static_cast<CCVS*>(e);
                    out << "  " << d->name << ": CCVS " << d->node1 << "-" << d->node2 << ", controlling source " << d->vName << ", gain=" << d->gain << "\n";
                    break;
                }
                    // CCCS (F)
                case DEP_CCCS: {
                    auto d = static_cast<CCCS*>(e);
                    out << "  " << d->name << ": CCCS " << d->node1 << "-" << d->node2 << ", controlling source " << d->vName << ", gain=" << d->gain << "\n";
                    break;
                }
            }
        }
    }

    //----------------------------------------------------------------------------------------
    //   DC solve (Modified Nodal Analysis + Newton‐Raphson for diodes)
    //
    //   We build an (N + M)×(N + M) matrix:
    //     – N non‐zero nodes
    //     – M “extra” unknowns for each voltage source and inductor (we no longer count GROUND here)
    //   We stamp:
    //     – Resistors: G = 1/R into G‐matrix
    //     – Capacitors: open‐circuit at DC (no stamping)
    //     – Inductors: DC short → they become “branch current” unknowns
    //     – Voltage sources: ideal → KCL/KVL stamp
    //     – Current sources: inject into RHS
    //     – Ground: fix node voltage to zero (no extra branch)
    //     – Diodes: in each Newton iteration, stamp a linearized conductance Gd = dI/dV and a constant Ieq
    //
    //   After convergence, we return node voltages in a vector indexed by the actual node number (0..maxNode).
    //----------------------------------------------------------------------------------------
    bool solveDC(vector<double>& nodeVoltages, ofstream& out) {
        // ► Page 38: No ground node detected in circuit
        if (nodeIndex.find(0) == nodeIndex.end()) {
            out << "Error: No ground node detected in circuit\n";
            return false;
        }

        int N = nodeCount;  // number of non‐zero nodes

        // count how many “extra branches”: each VSOURCE and INDUCTOR → one extra unknown
        int M = 0;
        for (auto e : elements) {
            if (e->type == VSOURCE || e->type == INDUCTOR)
                ++M;
        }
        int dim = N + M;
        if (dim == 0) {
            // no unknowns → trivial
            nodeVoltages.clear();
            return true;
        }

        // map each VSOURCE/INDUCTOR to a “branch index” in [0..M−1]
        map<CircuitElement*,int> branchIndex;
        int bi = 0;
        for (auto e : elements) {
            if (e->type == VSOURCE || e->type == INDUCTOR) {
                branchIndex[e] = bi++;
            }
        }

        // Prepare A and b
        vector<vector<double>> A(dim, vector<double>(dim, 0.0));
        vector<double> b(dim, 0.0);

        // Helper lambda to return “nodeIndex or –1 if ground”
        auto idxN = [&](int node) {
            return (node == 0 ? -1 : nodeIndex[node]);
        };

        // FIRST: stamp all linear parts (resistors, inductors as shorts, voltage & current sources, ground),
        // without diodes.  We will re‐stamp diodes inside Newton.
        for (auto e : elements) {
            int n1 = e->node1, n2 = e->node2;
            int i1 = idxN(n1), i2 = idxN(n2);

            switch (e->type) {
                case RESISTOR: {
                    double G = 1.0 / static_cast<Resistor*>(e)->resistance;
                    if (i1 >= 0) A[i1][i1] += G;
                    if (i2 >= 0) A[i2][i2] += G;
                    if (i1 >= 0 && i2 >= 0) {
                        A[i1][i2] -= G;
                        A[i2][i1] -= G;
                    }
                    break;
                }
                case CAPACITOR:
                    // open at DC → do nothing
                    break;

                case INDUCTOR: {
                    // treat as short: KCL: i_branch flows from n1→n2
                    int idxB = branchIndex[e];
                    // row/col N+idxB ↔ node voltages
                    if (i1 >= 0) {
                        A[i1][N + idxB] += 1.0;
                        A[N + idxB][i1] += 1.0;
                    }
                    if (i2 >= 0) {
                        A[i2][N + idxB] -= 1.0;
                        A[N + idxB][i2] -= 1.0;
                    }
                    b[N + idxB] = 0.0;  // no independent source for pure short
                    break;
                }

                case VSOURCE: {
                    int idxB = branchIndex[e];
                    double V = static_cast<VoltageSource*>(e)->voltage;
                    if (i1 >= 0) {
                        A[i1][N + idxB] += 1.0;
                        A[N + idxB][i1] += 1.0;
                    }
                    if (i2 >= 0) {
                        A[i2][N + idxB] -= 1.0;
                        A[N + idxB][i2] -= 1.0;
                    }
                    b[N + idxB] = V;
                    break;
                }

                case ISOURCE: {
                    double I = static_cast<CurrentSource*>(e)->current;
                    if (i1 >= 0) b[i1] -= I;
                    if (i2 >= 0) b[i2] += I;
                    break;
                }

                case GROUND: {
                    // Fix node voltage to zero: clear row & column, then clamp V=0
                    if (i1 >= 0) {
                        // zero the entire row and col first
                        for (int k = 0; k < dim; ++k) {
                            A[i1][k] = 0.0;
                            A[k][i1] = 0.0;
                        }
                        A[i1][i1] = 1.0;
                        b[i1] = 0.0;
                    }
                    break;
                }

                case DIODE:
                    // skip here; will be done inside Newton iterations
                    break;
            }
        }

        // Check if any diode exists
        bool hasDiode = false;
        for (auto e : elements) {
            if (e->type == DIODE) { hasDiode = true; break; }
        }

        // If no diodes, we can solve in one shot
        vector<double> x(dim, 0.0);
        if (!hasDiode) {
            // Just solve A x = b
            vector<vector<double>> A_copy = A;
            vector<double> b_copy = b;
            if (!GaussianSolver::solve(A_copy, b_copy, x)) {
                out << "Error: DC solve failed (singular matrix)\n";
                return false;
            }
        }
        else {
            // Newton‐Raphson loop
            const int maxIter = 50;
            const double tol = 1e-8;

            // Start with a guess x = 0
            for (int i = 0; i < dim; ++i) x[i] = 0.0;

            for (int iter = 0; iter < maxIter; ++iter) {
                // Rebuild A_nl and b_nl for this iteration
                // Start by copying all linear stamps:
                vector<vector<double>> A_nl = A;
                vector<double> b_nl = b;

                // Stamp each diode’s linearization
                for (auto e : elements) {
                    if (e->type != DIODE) continue;
                    auto d = static_cast<Diode*>(e);

                    int n1 = d->node1, n2 = d->node2;
                    int i1 = idxN(n1), i2 = idxN(n2);

                    // diode voltage = Vd = V(n1) – V(n2)
                    double Va = (i1 >= 0 ? x[i1] : 0.0);
                    double Vb = (i2 >= 0 ? x[i2] : 0.0);
                    double Vd = Va - Vb;

                    // I_diode = Is ( exp(Vd/(nVt)) – 1 )
                    double expo = exp(Vd / (d->emissionCoeff * d->thermalVoltage));
                    double Id = d->saturationCurrent * (expo - 1.0);

                    // conductance Gd = dI/dV = (Is/(nVt)) e^(Vd/(nVt))
                    double Gd = (d->saturationCurrent / (d->emissionCoeff * d->thermalVoltage)) * expo;

                    // Equivalent current source Ieq = Id – Gd*Vd
                    double Ieq = Id - Gd * Vd;

                    // stamp Gd into A_nl at (i1,i1), (i1,i2), (i2,i1), (i2,i2)
                    if (i1 >= 0) A_nl[i1][i1] += Gd;
                    if (i2 >= 0) A_nl[i2][i2] += Gd;
                    if (i1 >= 0 && i2 >= 0) {
                        A_nl[i1][i2] -= Gd;
                        A_nl[i2][i1] -= Gd;
                    }
                    // stamp Ieq into b_nl (subtract at node1, add at node2)
                    if (i1 >= 0) b_nl[i1] -= Ieq;
                    if (i2 >= 0) b_nl[i2] += Ieq;
                }

                // Solve A_nl x_new = b_nl
                vector<double> x_new(dim, 0.0);
                if (!GaussianSolver::solve(A_nl, b_nl, x_new)) {
                    out << "Warning: Newton iteration singular matrix\n";
                    break;
                }

                // Check convergence
                double maxDiff = 0.0;
                for (int i = 0; i < dim; ++i) {
                    maxDiff = max(maxDiff, fabs(x_new[i] - x[i]));
                    x[i] = x_new[i];
                }
                if (maxDiff < tol) break;
                if (iter == maxIter - 1) {
                    out << "Warning: Newton did not converge in DC solve\n";
                }
            }
        }

        // At this point, x[0..N−1] = node voltages at each non‐zero node index
        // We want to produce a vector nodeVoltages[0..maxNode], where index=actual node #
        int maxNode = 0;
        for (auto& kv : nodeIndex) {
            maxNode = max(maxNode, kv.first);
        }
        nodeVoltages.assign(maxNode+1, 0.0);
        nodeVoltages[0] = 0.0; // ground
        for (auto& kv : nodeIndex) {
            int actualNode = kv.first;
            int idx = kv.second;   // 0..N−1
            nodeVoltages[actualNode] = x[idx];
        }
        return true;
    }

    //----------------------------------------------------------------------------------------
    //   Transient simulation (Backward Euler or Trapezoidal halfway).  We do not re‐solve
    //   diode nonlinearities at each time‐step; instead, we freeze the diode conductance
    //   and equivalent current at its DC operating point and reuse it for the entire transient.
    //
    //   We print “Time   V(node1)   V(node2) …” for every time step.
    //----------------------------------------------------------------------------------------
    void simulateTransient(double totalTime, double timeStep, IntegrationMethod method, ofstream& out) {
        // Step 1: Do a DC solve once to fix diode Gd and Ieq at DC O.P.  Otherwise they might lock in the wrong branch.
        vector<double> dcVolt;
        if (!solveDC(dcVolt, out)) {
            out << "Warning: cannot get diode operating point—transient will crash\n";
        }

        // Build maps: each VSOURCE/INDUCTOR → branch index
        int N = nodeCount;
        map<CircuitElement*,int> branchIndex;
        int M = 0;
        for (auto e : elements) {
            if (e->type == VSOURCE || e->type == INDUCTOR) {
                branchIndex[e] = M++;
            }
        }
        int dim = N + M;

        // Prepare “frozen” diode conductances & Ieq from the DC operating point:
        // So that in each time‐step we simply add Gd and Ieq once.
        struct FrozenDiode { int i1, i2; double Gd, Ieq; };
        vector<FrozenDiode> frozenDiodes;
        for (auto e : elements) {
            if (e->type != DIODE) continue;
            auto d = static_cast<Diode*>(e);
            int i1 = (d->node1==0 ? -1 : nodeIndex[d->node1]);
            int i2 = (d->node2==0 ? -1 : nodeIndex[d->node2]);
            double Va = (i1>=0 ? dcVolt[d->node1] : 0.0);
            double Vb = (i2>=0 ? dcVolt[d->node2] : 0.0);
            double Vd = Va - Vb;
            double expo = exp(Vd / (d->emissionCoeff * d->thermalVoltage));
            double Id = d->saturationCurrent * (expo - 1.0);
            double Gd = (d->saturationCurrent / (d->emissionCoeff * d->thermalVoltage)) * expo;
            double Ieq = Id - Gd * Vd;
            frozenDiodes.push_back({ i1, i2, Gd, Ieq });
        }

        // Build a sorted list of actual node numbers to print in ascending order:
        vector<int> sortedNodes;
        for (auto& kv : nodeIndex) {
            sortedNodes.push_back(kv.first);
        }
        sort(sortedNodes.begin(), sortedNodes.end());

        // Print header
        out << "Time";
        for (int node : sortedNodes) {
            out << "\tV(" << node << ")";
        }
        out << "\n";

        // Time‐step loop
        int steps = int(ceil(totalTime / timeStep));
        double t = 0.0;
        for (int step = 0; step <= steps; ++step) {
            // Build MNA matrix & RHS
            vector<vector<double>> A(dim, vector<double>(dim, 0.0));
            vector<double> b(dim, 0.0);

            auto idxN = [&](int node)->int { return (node==0 ? -1 : nodeIndex[node]); };

            // 1) Stamp all linear elements just like in DC
            for (auto e : elements) {
                int n1 = e->node1, n2 = e->node2;
                int i1 = idxN(n1), i2 = idxN(n2);

                switch (e->type) {
                    case RESISTOR: {
                        double G = 1.0 / static_cast<Resistor*>(e)->resistance;
                        if (i1>=0) A[i1][i1] += G;
                        if (i2>=0) A[i2][i2] += G;
                        if (i1>=0 && i2>=0) {
                            A[i1][i2] -= G;
                            A[i2][i1] -= G;
                        }
                        break;
                    }
                    case CAPACITOR: {
                        auto c = static_cast<Capacitor*>(e);
                        double C = c->capacitance;
                        if (method == BACKWARD_EULER) {
                            double Gc = C / timeStep;
                            if (i1>=0) A[i1][i1] += Gc;
                            if (i2>=0) A[i2][i2] += Gc;
                            if (i1>=0 && i2>=0) {
                                A[i1][i2] -= Gc;
                                A[i2][i1] -= Gc;
                            }
                            double Ieq = Gc * c->lastVoltageDiff;
                            if (i1>=0) b[i1] -= Ieq;
                            if (i2>=0) b[i2] += Ieq;
                        } else {
                            // Trapezoidal rule
                            double Gc = 2.0 * C / timeStep;
                            if (i1>=0) A[i1][i1] += Gc;
                            if (i2>=0) A[i2][i2] += Gc;
                            if (i1>=0 && i2>=0) {
                                A[i1][i2] -= Gc;
                                A[i2][i1] -= Gc;
                            }
                            double Ieq = c->lastCurrent + Gc * c->lastVoltageDiff;
                            if (i1>=0) b[i1] -= Ieq;
                            if (i2>=0) b[i2] += Ieq;
                        }
                        break;
                    }
                    case INDUCTOR: {
                        auto l = static_cast<Inductor*>(e);
                        int idxB = branchIndex[e];
                        if (method == BACKWARD_EULER) {
                            double coeff = -l->inductance / timeStep;
                            if (i1>=0) {
                                A[i1][N+idxB] += 1.0;
                                A[N+idxB][i1] += 1.0;
                            }
                            if (i2>=0) {
                                A[i2][N+idxB] -= 1.0;
                                A[N+idxB][i2] -= 1.0;
                            }
                            A[N+idxB][N+idxB] += coeff;
                            b[N+idxB] = -coeff * l->lastCurrent;
                        } else {
                            // Trapezoidal
                            double coeff = -2.0 * l->inductance / timeStep;
                            if (i1>=0) {
                                A[i1][N+idxB] += 1.0;
                                A[N+idxB][i1] += 1.0;
                            }
                            if (i2>=0) {
                                A[i2][N+idxB] -= 1.0;
                                A[N+idxB][i2] -= 1.0;
                            }
                            A[N+idxB][N+idxB] += coeff;
                            b[N+idxB] = -(2.0 * l->inductance / timeStep * l->lastCurrent + l->lastVoltage);
                        }
                        break;
                    }
                    case DIODE:
                        // in this transient version, we “freeze” each diode’s Gd & Ieq from DC above
                        break;
                    case VSOURCE: {
                        int idxB = branchIndex[e];
                        double V = static_cast<VoltageSource*>(e)->voltage;
                        if (i1>=0) {
                            A[i1][N+idxB] += 1.0;
                            A[N+idxB][i1] += 1.0;
                        }
                        if (i2>=0) {
                            A[i2][N+idxB] -= 1.0;
                            A[N+idxB][i2] -= 1.0;
                        }
                        b[N+idxB] = V;
                        break;
                    }
                    case ISOURCE: {
                        double I = static_cast<CurrentSource*>(e)->current;
                        if (i1>=0) b[i1] -= I;
                        if (i2>=0) b[i2] += I;
                        break;
                    }
                    case GROUND: {
                        // fix node voltage to zero: clear row & column
                        if (i1 >= 0) {
                            for (int k = 0; k < dim; ++k) {
                                A[i1][k] = 0.0;
                                A[k][i1] = 0.0;
                            }
                            A[i1][i1] = 1.0;
                            b[i1] = 0.0;
                        }
                        break;
                    }
                }
            }

            // 2) Stamp frozen diode linearization from DC into A & b
            for (auto& fd : frozenDiodes) {
                int i1 = fd.i1, i2 = fd.i2;
                double Gd = fd.Gd, Ieq = fd.Ieq;
                if (i1>=0) A[i1][i1] += Gd;
                if (i2>=0) A[i2][i2] += Gd;
                if (i1>=0 && i2>=0) {
                    A[i1][i2] -= Gd;
                    A[i2][i1] -= Gd;
                }
                if (i1>=0) b[i1] -= Ieq;
                if (i2>=0) b[i2] += Ieq;
            }

            // Solve A x = b
            vector<double> x(dim, 0.0);
            if (!GaussianSolver::solve(A, b, x)) {
                out << "Warning: Transient solve failed at t=" << t << "\n";
            }

            // Write voltages
            out << t;
            for (int node : sortedNodes) {
                int idx = nodeIndex[node];
                double Vn = x[idx];
                out << "\t" << Vn;
            }
            out << "\n";

            // Update states for capacitors & inductors (for next time step)
            if (step < steps) {
                for (auto e : elements) {
                    if (e->type == CAPACITOR) {
                        auto c = static_cast<Capacitor*>(e);
                        int i1 = idxN(c->node1), i2 = idxN(c->node2);
                        double Va = (i1>=0 ? x[i1] : 0.0);
                        double Vb = (i2>=0 ? x[i2] : 0.0);
                        double Vdiff_new = Va - Vb;
                        if (method == BACKWARD_EULER) {
                            c->lastVoltageDiff = Vdiff_new;
                        } else {
                            double newI = 2.0 * c->capacitance / timeStep * (Vdiff_new - c->lastVoltageDiff)
                                          - c->lastCurrent;
                            c->lastCurrent = newI;
                            c->lastVoltageDiff = Vdiff_new;
                        }
                    }
                    else if (e->type == INDUCTOR) {
                        auto l = static_cast<Inductor*>(e);
                        int idxB = branchIndex[e];
                        double i_new = x[N + idxB];
                        int i1 = idxN(l->node1), i2 = idxN(l->node2);
                        double Va = (i1>=0 ? x[i1] : 0.0);
                        double Vb = (i2>=0 ? x[i2] : 0.0);
                        double Vdiff_new = Va - Vb;
                        l->lastCurrent = i_new;
                        l->lastVoltage = Vdiff_new;
                    }
                }
            }

            t += timeStep;
        }
    }
};

//-----------------------------------------------------------------------------
//   Simple string utilities (trim + tokenize by whitespace)
//-----------------------------------------------------------------------------
static inline string trim(const string& s) {
    size_t i = s.find_first_not_of(" \t\r\n");
    if (i == string::npos) return "";
    size_t j = s.find_last_not_of(" \t\r\n");
    return s.substr(i, j - i + 1);
}

static inline vector<string> tokenize(const string& line) {
    vector<string> tok;
    istringstream iss(line);
    string w;
    while (iss >> w) tok.push_back(w);
    return tok;
}

int parseNode(const string& nodeStr) {
    if (nodeStr == "GND") return 0;
    string numStr;
    for (char c : nodeStr) {
        if (isdigit(c)) numStr += c;
    }

    if (numStr.empty()) return 0; // Default to ground if no digits found
    try {
        return stoi(numStr);
    } catch (...) {
        return 0;
    }
}

double convertValue(const string& valStr) {
    if (valStr.empty()) return 0.0;
    char last = valStr.back();
    double multiplier = 1.0;
    string numPart = valStr;

    if (!isdigit(last)) {
        numPart = valStr.substr(0, valStr.size()-1);
        switch (tolower(last)) {
            case 't': multiplier = 1e12; break;
            case 'g': multiplier = 1e9;  break;
            case 'k': multiplier = 1e3;  break;
            case 'm':
                if (valStr.size() >= 3 && valStr.substr(valStr.size()-3) == "MEG") {
                    multiplier = 1e6;
                    numPart = valStr.substr(0, valStr.size()-3);
                } else {
                    multiplier = 1e6;
                }
                break;
            case 'h': multiplier = 1e2;  break;
            case 'd': multiplier = 1e-1; break;
            case 'c': multiplier = 1e-2; break;
            case 'u': multiplier = 1e-6; break;
            case 'n': multiplier = 1e-9; break;
            case 'p': multiplier = 1e-12;break;
            case 'f': multiplier = 1e-15;break;
            default:  multiplier = 1.0;  break;
        }
    }

    try {
        // 1) compute numeric value
        double value = stod(numPart) * multiplier;
        // 2) check for non‐positive
        if (value <= 0.0) {
            throw ValueException("Error: Component value must be positive");
        }
        // 3) return if OK
        return value;
    }
    catch (const ValueException&) {
        // pass through our positive‐value error
        throw;
    }
    catch (...) {
        // any other parse error
        throw SyntaxException();
    }
}

//void Circuit::saveToFile(const string& filename) {
//    ofstream file(schematicsDir + filename);
//    for (auto e : elements) {
//        switch (e->type) {
//            case RESISTOR: {
//                auto r = static_cast<Resistor*>(e);
//                out << "  " << r->name << ": Resistor "
//                    << r->node1 << "-" << r->node2
//                    << ", " << r->resistance << " Ohm\n";
//                break;
//            }
//            case CAPACITOR: {
//                auto c = static_cast<Capacitor*>(e);
//                out << "  " << c->name << ": Capacitor "
//                    << c->node1 << "-" << c->node2
//                    << ", " << c->capacitance << " F\n";
//                break;
//            }
//            case INDUCTOR: {
//                auto l = static_cast<Inductor*>(e);
//                out << "  " << l->name << ": Inductor "
//                    << l->node1 << "-" << l->node2
//                    << ", " << l->inductance << " H\n";
//                break;
//            }
//            case DIODE: {
//                auto d = static_cast<Diode*>(e);
//                out << "  " << d->name << ": Diode "
//                    << d->node1 << "->" << d->node2
//                    << ", Is=" << d->saturationCurrent
//                    << ", n=" << d->emissionCoeff
//                    << ", Vt=" << d->thermalVoltage << " V\n";
//                break;
//            }
//            case VSOURCE: {
//                auto v = static_cast<VoltageSource*>(e);
//                out << "  " << v->name << ": Vsrc "
//                    << v->node1 << "-" << v->node2
//                    << " = " << v->voltage << " V\n";
//                break;
//            }
//            case ISOURCE: {
//                auto i = static_cast<CurrentSource*>(e);
//                out << "  " << i->name << ": Isrc "
//                    << i->node1 << "->" << i->node2
//                    << " = " << i->current << " A\n";
//                break;
//            }
//            case GROUND: {
//                auto g = static_cast<Ground*>(e);
//                out << "  " << g->name << ": Ground at node "
//                    << g->node1 << "\n";
//                break;
//            }
//                // VCVS (E)
//            case DEP_VCVS: {
//                auto d = static_cast<VCVS*>(e);
//                out << "  " << d->name << ": VCVS " << d->node1 << "-" << d->node2 << ", ctrl nodes " << d->ctrlNode1 << "-" << d->ctrlNode2 << ", gain=" << d->gain << "\n";
//                break;
//            }
//                // VCCS (G)
//            case DEP_VCCS: {
//                auto d = static_cast<VCCS*>(e);
//                out << "  " << d->name << ": VCCS " << d->node1 << "-" << d->node2 << ", ctrl nodes " << d->ctrlNode1 << "-" << d->ctrlNode2 << ", gain=" << d->gain << "\n";
//                break;
//            }
//                // CCVS (H)
//            case DEP_CCVS: {
//                auto d = static_cast<CCVS*>(e);
//                out << "  " << d->name << ": CCVS " << d->node1 << "-" << d->node2 << ", controlling source " << d->vName << ", gain=" << d->gain << "\n";
//                break;
//            }
//                // CCCS (F)
//            case DEP_CCCS: {
//                auto d = static_cast<CCCS*>(e);
//                out << "  " << d->name << ": CCCS " << d->node1 << "-" << d->node2 << ", controlling source " << d->vName << ", gain=" << d->gain << "\n";
//                break;
//            }
//        }
//    }
//    file << ".end\n";
//    file.close();
//}

//-----------------------------------------------------------------------------
//   main(): read “menu” from input, build circuit, respond to commands, write to output.
//-----------------------------------------------------------------------------
int main() {
    string inputFilename;
    ofstream archiveOut;
    bool isNew = false;
    // 1) Open input and output files
    struct stat st = {0};
    if (stat(schematicsDir.c_str(), &st) == -1) {
        mkdir(schematicsDir.c_str());
    }
    ifstream fin(inputPath);
    ofstream fout(outputPath);
    if (!fin.is_open() || !fout.is_open()) {
        cerr << "Cannot open input or output file\n";
        return 1;
    }

    // 2) Build list of .txt in ./schematics
    vector<string> availableFiles;
    DIR* dir = opendir("schematics");
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            string f = entry->d_name;
            if (f.size() >= 4 && f.substr(f.size()-4) == ".txt") {
                availableFiles.push_back(f);
            }
        }
        closedir(dir);
    }

    // 3) Print menu
    fout << "– File Menu –\n";
    fout << "1) show existing schematics\n";
    fout << "2) load schematic from file\n";
    fout << "3) new schematic\n";
    fout << "4) exit\n";
    fout << "Enter choice:\n";

    // 4) Read choice
    int choice = 0;
    fin >> choice;
    if (!fin.good() || choice < 1 || choice > 4) {
        fout << "Error: Inappropriate input\n";
        return 0;
    }

    string filename;

    if (choice == 1) {
        // Option 1: Show existing schematics
        fout << "- choose existing schematic:\n";
        if (availableFiles.empty()) {
            fout << "Error: No schematic files found\n";
            return 0;
        }
        for (int i = 0; i < (int)availableFiles.size(); ++i) {
            fout << (i + 1) << ") " << availableFiles[i] << "\n";
        }
        fout << "Enter number or filename:\n";
        string choiceStr;
        fin >> choiceStr;
        if (!fin.good()) {
            fout << "Error: Inappropriate input\n";
            return 0;
        }
        int sel = 0;
        try {
            sel = stoi(choiceStr);
            if (sel < 1 || sel > (int)availableFiles.size()) {
                throw out_of_range("");
            }
            filename = availableFiles[sel - 1];
        } catch (...) {
            auto it = find(availableFiles.begin(), availableFiles.end(), choiceStr);
            if (it == availableFiles.end()) {
                fout << "Error: Inappropriate input\n";
                return 0;
            }
            filename = *it;
        }
    }
    else if (choice == 2) {
        // Option 2: prompt for filename
        fout << "Enter netlist filename:\n";
        fin >> filename;
        if (!fin.good()) {
            fout << "Error: Inappropriate input\n";
            return 0;
        }
    }
    else if (choice == 3) {
        // Option 3: Create new schematic
        fout << "New schematic. Enter filename to create:\n";
        fin >> filename;
        if (!fin.good()) {
            fout << "Error: Inappropriate input\n";
            return 0;
        }
        // Create and register file
        ofstream ofs(schematicsDir + filename);
        ofs.close();
        isNew = true;
//        archiveOut.open(schematicsDir + filename, ios::app);
    }
    else {
        // Option 4: exit
        return 0;
    }

    ifstream netin;

    // announce loading and start command loop
    fout << "Loading schematic: " << filename << "\n";
    fout << "Enter commands (type .end to finish):\n";
    fin.ignore(numeric_limits<streamsize>::max(), '\n');

    if (isNew) {
        archiveOut.open(schematicsDir + filename, ios::app);
        if (!archiveOut.is_open()) {
            fout << "Error: Cannot open archive file for writing\n";
            return 0;
        }
    }

    if (!isNew) {
        netin.open(schematicsDir + filename);
        if (!netin.is_open()) {
            fout << "Error: File not found\n";
            return 0;
        }
    }

    Circuit circuit;

    // Shared command-processing loop
    istream* pin = isNew ? &fin : &netin;
    ifstream fin2;

    string raw, line;
    while (getline(*pin, raw)) {
        // trim whitespace
        int a = raw.find_first_not_of(" \t\r\n");
        if (a == string::npos) continue;
        int b = raw.find_last_not_of(" \t\r\n");
        line = raw.substr(a, b - a + 1);

        // always archive the line if new schematic
        if (isNew)
            archiveOut << line << "\n";

        // stop on .end
        if (line == ".end")
            break;

        // tokenize
        istringstream iss(line);
        vector<string> tok;
        string w;
        while (iss >> w) tok.push_back(w);
        if (tok.empty()) continue;

        // dispatch commands inline
        if (tok[0] == "add") {
            try {
                if (tok.size() < 3) throw SyntaxException();
                string name = tok[1]; char c = name[0];
                // Ground
                if (c=='G' && tok.size()==3) {
                    int n1=parseNode(tok[2]);
                    circuit.addGround(name,n1);
//                    circuit.saveToFile(schematicsDir + filename);
                }
                    // VCVS
                else if (c=='E' && tok.size()==7) {
                    int n1=parseNode(tok[2]),n2=parseNode(tok[3]);
                    int cn1=parseNode(tok[4]),cn2=parseNode(tok[5]);
                    double g=convertValue(tok[6]);
                    circuit.addVCVS(name,n1,n2,cn1,cn2,g);
//                    circuit.saveToFile(schematicsDir + filename);
                }
                    // VCCS
                else if (c=='G' && tok.size()==7) {
                    int n1=parseNode(tok[2]),n2=parseNode(tok[3]);
                    int cn1=parseNode(tok[4]),cn2=parseNode(tok[5]);
                    double g=convertValue(tok[6]);
                    circuit.addVCCS(name,n1,n2,cn1,cn2,g);
//                    circuit.saveToFile(schematicsDir + filename);
                }
                    // CCVS
                else if (c=='H' && tok.size()==6) {
                    int n1=parseNode(tok[2]),n2=parseNode(tok[3]);
                    string vs=tok[4]; double g=convertValue(tok[5]);
                    circuit.addCCVS(name,n1,n2,vs,g);
//                    circuit.saveToFile(schematicsDir + filename);
                }
                    // CCCS
                else if (c=='F' && tok.size()==6) {
                    int n1=parseNode(tok[2]),n2=parseNode(tok[3]);
                    string vs=tok[4]; double g=convertValue(tok[5]);
                    circuit.addCCCS(name,n1,n2,vs,g);
//                    circuit.saveToFile(schematicsDir + filename);
                }
                    // R,C,L,V,I
                else if ((c=='R'||c=='C'||c=='L'||c=='V'||c=='I') && tok.size()==5) {
                    int n1=parseNode(tok[2]),n2=parseNode(tok[3]);
                    double v=convertValue(tok[4]);
                    if (c=='R') {
                        circuit.addResistor(name,n1,n2,v);
//                        circuit.saveToFile(schematicsDir + filename);
                    }
                    else if (c=='C') {
                        circuit.addCapacitor(name,n1,n2,v);
//                        circuit.saveToFile(schematicsDir + filename);
                    }
                    else if (c=='L') {
                        circuit.addInductor(name,n1,n2,v);
//                        circuit.saveToFile(schematicsDir + filename);
                    }
                    else if (c=='V') {
                        circuit.addVoltageSource(name,n1,n2,v);
//                        circuit.saveToFile(schematicsDir + filename);
                    }
                    else {circuit.addCurrentSource(name,n1,n2,v);
//                        circuit.saveToFile(schematicsDir + filename);
                    }
                }
                    // Diode
                else if (c=='D' && (tok.size()==4||tok.size()==5)) {
                    string m=(tok.size()==5?tok[4]:"D");
                    if (m!="D"&&m!="Z") throw ModelException();
                    double Is=(m=="D"?1e-14:1e-12),nc=(m=="D"?1.0:1.2),Vt=0.02585;
                    int n1=parseNode(tok[2]),n2=parseNode(tok[3]);
                    circuit.addDiode(name,n1,n2,Is,nc,Vt);
//                    circuit.saveToFile(schematicsDir + filename);
                }
                else throw NameException();
            }
            catch(const SyntaxException&) {
                fout<<"Error: Inappropriate input\n";
            }
            catch(const ValueException&) {
                fout<<"Error: Component value must be positive\n";
            }
            catch(const DuplicateException& e) {
                fout<<e.what()<<"\n";
            }
            catch(const NameException& e) {
                fout<<e.what()<<"\n";
            }
            catch(const exception& e) {
                fout<<e.what()<<"\n";
            }
        }
        else if (tok[0]=="delete") {
            try {
                if(tok.size()!=2) throw SyntaxException();
                circuit.deleteElement(tok[1]);
//                circuit.saveToFile(schematicsDir + filename);
            } catch(const exception& e) {
                fout<<e.what()<<"\n";
            }
        }
        else if (tok[0]=="list") {
            circuit.listElements(fout);
        }
        else if (tok[0]=="solve" && tok.size()==2 && tok[1]=="dc") {
            vector<double> nv;
            circuit.solveDC(nv,fout);
        }
        else if (tok[0]=="tran" && tok.size()==4) {
            try {
                double ts=stod(tok[1]),st=stod(tok[2]),sp=stod(tok[3]);
                circuit.simulateTransient(sp,ts,BACKWARD_EULER,fout);
            } catch(...) {
                fout<<"Error: Syntax error in tran command\n";
            }
        }
        else {
            fout<<"Error: Syntax error\n";
        }
    }

    if (isNew) archiveOut.close();
    if (!isNew) netin.close();
    fout<<"Exiting.\n";
    return 0;
}
