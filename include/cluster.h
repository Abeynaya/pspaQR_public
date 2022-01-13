#ifndef CLUSTER_H
#define CLUSTER_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <list>
#include <set>
#include <tuple>
#include <assert.h>

#include "util.h"

typedef Eigen::SparseMatrix<double, 0, int> SpMat;

struct Edge;

/*Separator ID: gives level and separator index
  in that level
*/
struct SepID {
    public: 
        int lvl;
        int sep;
        SepID(int l, int s) : lvl(l), sep(s){};
        SepID() : lvl(-1), sep(0){};

        int level() const {return lvl;}
        int index() const {return sep;}
        // Some lexicographics order
        // NOT the matrix ordering
        bool operator==(const SepID& other) const {
            return (this->lvl == other.lvl && this->sep == other.sep);
        }
        bool operator<(const SepID& other) const {
            return (this->lvl < other.lvl) 
                    || (this->lvl == other.lvl && this->sep < other.sep);
        }
        bool operator!=(const SepID& other) const {
            return (this->lvl != other.lvl  || this->sep != other.sep );
        }
};

// Describes the merging of the separators
struct ClusterID {
    public:
        SepID self;
        SepID l;
        SepID r;
        int part; // left or right part of the separator
        int section; // which partition of the recursive bissection process
        
        ClusterID(SepID self) {
            this->self = self;
            this->l    = SepID();
            this->r    = SepID();
            this->part = 2;
            this->section = 0;
        }

        ClusterID() {
            this->self = SepID();
            this->l    = SepID();
            this->r    = SepID();
            this->part = 2;
            this->section = 0;
        }

        ClusterID(SepID self, SepID left, SepID right) {
            this->self = self;
            this->l    = left;
            this->r    = right;
            this->part = 2;
            this->section = 0;
        }

        ClusterID(SepID self, SepID left, SepID right, int part) {
            this->self = self;
            this->l    = left;
            this->r    = right;
            this->part = part;
            this->section = 0;
        }


        ClusterID(SepID self, SepID left, SepID right, int part, int section) {
            this->self = self;
            this->l    = left;
            this->r    = right;
            this->part = part;
            this->section = section;
        }

        int level() const {return self.lvl;};
        SepID left() const {return l;};
        SepID right() const {return r;};



        // Some lexicographics order
        // NOT the matrix ordering
        bool operator==(const ClusterID& other) const {
            return      (this->self == other.self)
                     && (this->l    == other.l)
                     && (this->r    == other.r)
                     && (this->part == other.part)
                     && (this->section == other.section);
        }
        bool operator!=(const ClusterID& other) const{
            return !(*this==other);
        }
        bool operator<(const ClusterID& other) const {
            return     (this->self <  other.self)
                    || (this->self == other.self && this->l <  other.l) 
                    || (this->self == other.self && this->l == other.l && this->r < other.r)
                    || (this->self == other.self && this->l == other.l && this->r == other.r && this->section < other.section)
                    || (this->self == other.self && this->l == other.l && this->r == other.r && this->section == other.section && this->part < other.part);
        }
};

std::ostream& operator<<(std::ostream& os, const SepID& s);
std::ostream& operator<<(std::ostream& os, const ClusterID& c);
SepID merge(SepID& s);
ClusterID merge_if(ClusterID& c, int lvl);

struct Cluster{
private:
    int cstart; // start != -1 => botttom level in the cluster heirarchy 
    int csize; // size of cluster columns
    int rstart;
    int rsize; // number of rows assosciated with the cluster
    ClusterID id; 
    int order; // Elimination order
    bool eliminated;
    int rank; // to store the rank after sparsification

    // Original size of the cluster (when the cluster is formed)
    int csize_org;
    int rsize_org;

    /* Hierarchy */
    Cluster* parent; 
    ClusterID parentid;

    /* Self edge for easy access -- all clusters will have this */
    Edge* eself = nullptr; 

    /* Matrices stored from factorization */
    /* Elmn */
    Eigen::MatrixXd* Q = nullptr;
    Eigen::VectorXd* tau = nullptr;
    Eigen::MatrixXd* T = nullptr;
    /* Scaling */
    Eigen::MatrixXd* Qs = nullptr;
    Eigen::MatrixXd* Ts = nullptr; 
    Eigen::VectorXd* taus = nullptr;
    /* Sparsification */
    Eigen::MatrixXd* Q_sp = nullptr;
    Eigen::MatrixXd* T_sp = nullptr; 
    Eigen::VectorXd* tau_sp = nullptr;


    /* Solution to linear system*/
    Eigen::VectorXd* x = nullptr; // Ax = b


public: 
    int rposparent; // row position in the parent
    int cposparent; // column position in the parent
    std::vector<Cluster*> children;
    std::set<Cluster*> rsparsity; 


    std::set<Cluster*> cnbrs; // neigbors including self
    std::set<Cluster*> rnbrs; // neigbors including self
    std::list<Edge*> edgesOut;
    std::list<Edge*> edgesIn;

    std::list<Edge*> edgesOutNbrSparsification;
    std::list<Edge*> edgesInNbrSparsification;

    /* Solution to A' xt=b */
    Eigen::VectorXd* xt = nullptr; // A'xt = b



public:
    /* Methods */
    Cluster(int cstart_, int csize_, int rstart_, int rsize_, ClusterID id_, int order_) :
            cstart(cstart_), csize(csize_), rstart(rstart_), rsize(rsize_), id(id_), order(order_), eliminated(false)
            {
                assert(rstart >= 0);
                set_size(rsize_, csize_);
                csize_org = csize_;
                rsize_org = rsize_;
                rank = csize_;
            };

    // Whether a cluster has been eliminated
    bool is_eliminated() const {
        return eliminated;
    }
    // Set clusters as eliminated
    void set_eliminated() {
       assert(! eliminated);
       eliminated = true;
    }

    bool operator==(const Cluster& other) const {
            return (this->id  == other.id && this->order == other.order);
    }

    bool operator!=(const Cluster& other) const {
        return (!(*this == other));
    }

    // 
    bool is_twin(Cluster* n); // is n the other part of the cluster

    int get_order() const; 
    int get_level() const; 
    int cols() const;
    int rows() const;
    void set_rank(int r);
    int get_rank() const;
    int get_cstart() const;
    int get_rstart() const;
    int original_rows() const;
    int original_cols() const;
    void set_org(int r, int c);

    ClusterID get_id () const;
    SepID get_sepID();

    int part();

    /*Heirarchy*/
    Cluster* get_parent();
    ClusterID get_parentid();
    void set_parentid(ClusterID cid);
    void set_parent(Cluster* p);
    void add_children(Cluster* c);

    /* Edges */
    Edge* self_edge();
    void add_self_edge(Edge*);

    void add_edgeOut(Edge* e);
    void add_edgeIn(Edge* e);
    void sort_edgesOut(bool reverse = false);

    /*Elimination*/
    void set_Q(Eigen::MatrixXd&);
    void set_tau(Eigen::VectorXd&);
    void set_T(Eigen::MatrixXd&);
    void set_tau(int, int);
    Eigen::VectorXd* get_tau();
    Eigen::MatrixXd* get_T();
    Eigen::MatrixXd* get_Q();
    void set_size(int r, int c);

    /* Scaling */
    void set_Qs(Eigen::MatrixXd&);
    void set_Ts(Eigen::MatrixXd&);
    void set_taus(Eigen::VectorXd&);
    Eigen::MatrixXd* get_Qs();
    Eigen::MatrixXd* get_Ts();
    Eigen::VectorXd* get_taus();

    /* Sparsification */
    void reset_size(int r, int c);
    void resize_x(int r);
    void add_edge_spars_out(Edge*);
    void add_edge_spars_in(Edge*);
    void set_Q_sp(Eigen::MatrixXd&);
    void set_T_sp(Eigen::MatrixXd&);
    void set_tau_sp(Eigen::VectorXd&);
    Eigen::MatrixXd* get_Q_sp();
    Eigen::MatrixXd* get_T_sp();
    Eigen::VectorXd* get_tau_sp();

    /* Solution to linear system */
    Segment head(); // return the first this->get_ncols() size of x
    Segment full(); // return the full x
    Segment extra(); // return part of x corresponding only to the extra rows
    Eigen::VectorXd* get_x();
    void set_vector(const Eigen::VectorXd& b);
    void set_vector_x(const Eigen::VectorXd& b);
    void tset_vector(const Eigen::VectorXd& b);
    Segment thead(); // return the first this->get_ncols() size of xt
    Segment tfull(); // return the full xt

    void extract_vector(Eigen::VectorXd& soln);
    void textract_vector(Eigen::VectorXd& soln);
    void textract_vector();



    /* Destructor */
    ~Cluster();

};

#endif