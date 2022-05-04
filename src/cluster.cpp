#include "cluster.h"
#include "edge.h"

typedef Eigen::SparseMatrix<double, 0, int> SpMat;

/* Print SepID */
std::ostream& operator<<(std::ostream& os, const SepID& s) {
    os << "(" << s.lvl <<  " " << s.sep << ")";
    return os;
}

/* Print ClusterID */
std::ostream& operator<<(std::ostream& os, const ClusterID& c) {
    os << "(" << c.self << "," << c.section << ":" << c.l << ";" << c.r << ")";
    return os;
}


SepID merge(SepID& s) {
    return SepID(s.lvl + 1, s.sep / 2);
}

ClusterID merge_if(ClusterID& c, int lvl) {
    auto left  = c.l.lvl < lvl ? merge(c.l) : c.l;
    auto right = c.r.lvl < lvl ? merge(c.r) : c.r;
    auto section = c.section;
    if (lvl > 0) 
        section /= 2;
    auto part = c.self.lvl == lvl ? 2 : c.part;

    return ClusterID(c.self, left, right, part, section);
}


int Cluster::get_order() const {return order;}
int Cluster::lvl() const {return id.self.lvl;} // Get level of the cluster
int Cluster::merge_lvl() const {return merge_level;}
int Cluster::cols() const {return csize;}
int Cluster::rows() const {return rsize;}
void Cluster::set_rank(int r){rank = r;}
void Cluster::set_row_rank(int r){row_rank = r; assert(r <= rsize-csize);}

int Cluster::get_rank() const {return rank;}
int Cluster::get_row_rank() const {return row_rank;}

int Cluster::get_cstart() const {return cstart;}
int Cluster::get_rstart() const {return rstart;}
int Cluster::original_rows() const {return rsize_org;};
int Cluster::original_cols() const {return csize_org;};
void Cluster::set_org(int r, int c) {rsize_org = r; csize_org = c;} 
ClusterID Cluster::get_id() const {return id;}
SepID Cluster::get_sepID(){return id.self;};
int Cluster::part(){return id.part;};
int Cluster::section(){return id.section;};
void Cluster::set_local_cstart(int c) {local_cstart = c;};
void Cluster::set_local_rstart(int r) {local_rstart = r;};
int Cluster::cstart_local() {return local_cstart;};
int Cluster::rstart_local() {return local_rstart;};

/*Heirarchy*/
Cluster* Cluster::get_parent(){return parent;}
ClusterID Cluster::get_parentid(){return parentid;}
void Cluster::set_parentid(ClusterID cid){parentid = cid;}
void Cluster::set_parent(Cluster* p){parent = p;}
void Cluster::add_children(Cluster* c){children.push_back(c);}
void Cluster::sort_children(){
    children.sort([](Cluster* c1, Cluster* c2){return c1->get_id() < c2->get_id();});
}

Edge* Cluster::self_edge(){
    assert(eself!=nullptr);
    return eself;
}

void Cluster::add_self_edge(Edge* e){
    eself = e;
}
void Cluster::add_edgeOut(Edge* e){edgesOut.push_back(e);}
void Cluster::add_edge_spars_out(Edge* e){edgesOutNbrSparsification.push_back(e);}

void Cluster::add_edgeOut_org(spEdge* e){edgesOut_org.push_back(e);}

void Cluster::add_edgeIn(Edge* e){edgesIn.push_back(e);}

void Cluster::add_edgeIn(Edge* e, bool is_spars_nbr){
    std::lock_guard<std::mutex> lock(mutex_edgeIn);
    edgesIn.push_back(e);
    if (is_spars_nbr) edgesInNbrSparsification.push_back(e);
    if (e->n1->lvl() == e->n1->merge_lvl()) col_interior_deps++;
}

void Cluster::sort_edgesOut(bool reverse){
    if (reverse){
        edgesOut.sort([](Edge* a, Edge* b){return a->n2->get_order() > b->n2->get_order();});
    }
    else {
        edgesOut.sort([](Edge* a, Edge* b){return a->n2->get_order() < b->n2->get_order();});
    }
}


EdgeIt Cluster::find_out_edge(int n_order) {
    auto found = find_if(edgesOut.begin(), edgesOut.end(), [&n_order](Edge* e){return e->n2->get_order() == n_order;});
    assert(found != edgesOut.end());
    return found;
}

/*Elimination*/
void Cluster::create_T(int norder) {
    this->Tmap[norder] = new Eigen::MatrixXd(0,0);
}
void Cluster::set_T(int norder, Eigen::MatrixXd& T_) {
    *(this->Tmap.at(norder)) = T_;
}

void Cluster::set_tau(int r, int c){
    int k = std::min(r,c);
    this->tau = new Eigen::VectorXd(k);
    this->T = new Eigen::MatrixXd(k,k);
    this->tau->setZero();
    this->T->setZero();
}

// Eigen::MatrixXd* Cluster::get_Q(){return this->Q;}
Eigen::VectorXd* Cluster::get_tau(){return this->tau;}
Eigen::MatrixXd* Cluster::get_T(){return this->T;}
Eigen::MatrixXd* Cluster::get_T(int norder){
    assert(this->Tmap.at(norder)->rows() != 0);
    return this->Tmap.at(norder);
}

/* Scaling */
void Cluster::set_Qs(Eigen::MatrixXd& Q_) {
    this->Qs = new Eigen::MatrixXd(0,0);
    *(this->Qs) = Q_;
}
void Cluster::set_Ts(Eigen::MatrixXd& T_) {
    this->Ts = new Eigen::MatrixXd(0,0);
    *(this->Ts) = T_;
}
void Cluster::set_taus(Eigen::VectorXd& t_) {
    this->taus = new Eigen::VectorXd(0);
    *(this->taus) = t_;
}

Eigen::MatrixXd* Cluster::get_Qs(){return this->Qs;}
Eigen::MatrixXd* Cluster::get_Ts(){return this->Ts;}
Eigen::VectorXd* Cluster::get_taus(){return this->taus;}

/* Reassign */
void Cluster::update_extra_rnorm(int n_order, Eigen::VectorXd& rnorm){
    std::lock_guard<std::mutex> lock(mutex_cluster);
    int extra_rows = this->rsize_org - this->csize_org;
    assert(this->r2c.size() == extra_rows);
    for (int i=0; i<extra_rows; ++i){
        if (rnorm(i)>std::get<1>(r2c[i])) {
            r2c[i] = std::make_tuple(n_order, rnorm(i));
        }
    }
}

void Cluster::add_extra_rows(int c_order, int istart, int nextra){
    std::lock_guard<std::mutex> lock(mutex_add_extra);
    extra_rows_from_cluster.push_back(std::make_tuple(c_order, istart, nextra));
    tot_extra += nextra;
}

void Cluster::sort_extra_rows_from_cluster(){
    std::lock_guard<std::mutex> lock(mutex_add_extra);  // safe to reuse mutex here
    if (extra_is_sorted) return;
    extra_rows_from_cluster.sort();
    extra_is_sorted = true;
}

/* Sparsification  */
void Cluster::set_Q_sp(Eigen::MatrixXd& Q_) {
    this->Q_sp = new Eigen::MatrixXd(0,0);
    *(this->Q_sp) = Q_;
}
void Cluster::set_T_sp(Eigen::MatrixXd& T_) {
    this->T_sp = new Eigen::MatrixXd(0,0);
    *(this->T_sp) = T_;
}
void Cluster::set_tau_sp(Eigen::VectorXd& t_) {
    this->tau_sp = new Eigen::VectorXd(0);
    *(this->tau_sp) = t_;
}

Eigen::MatrixXd* Cluster::get_Q_sp(){return this->Q_sp;}
Eigen::MatrixXd* Cluster::get_T_sp(){return this->T_sp;}
Eigen::VectorXd* Cluster::get_tau_sp(){return this->tau_sp;}

void Cluster::set_Q_sp_ex(Eigen::MatrixXd& Q_) {
    this->Q_sp_ex = new Eigen::MatrixXd(0,0);
    *(this->Q_sp_ex) = Q_;
}
void Cluster::set_T_sp_ex(Eigen::MatrixXd& T_) {
    this->T_sp_ex = new Eigen::MatrixXd(0,0);
    *(this->T_sp_ex) = T_;
}
void Cluster::set_tau_sp_ex(Eigen::VectorXd& t_) {
    this->tau_sp_ex = new Eigen::VectorXd(0);
    *(this->tau_sp_ex) = t_;
}

Eigen::MatrixXd* Cluster::get_Q_sp_ex(){return this->Q_sp_ex;}
Eigen::MatrixXd* Cluster::get_T_sp_ex(){return this->T_sp_ex;}
Eigen::VectorXd* Cluster::get_tau_sp_ex(){return this->tau_sp_ex;}


/**
 *  Solving the linear system 
**/

void Cluster::set_size(int r, int c){
    rsize = r;
    csize = c;
    rank = c;
    row_rank = r-c;
    if (merge_level == id.level()) {
        r2c.resize(rsize-csize, std::make_tuple(order, 0.0));
        row_perm  = Eigen::VectorXi::LinSpaced(rsize-csize, 0, rsize-csize-1);
    }
    delete this->x;
    this->x = new Eigen::VectorXd(r);
    this->x->setZero();

    delete this->xt;
    this->xt = new Eigen::VectorXd(r);
    this->xt->setZero();
}

void Cluster::reset_size(int r, int c){
    rsize = r;
    csize = c;
    rank = c;
    row_rank = r-c;
}

void Cluster::resize_x(int r){
    int old_r = this->x->size();
    this->x->conservativeResize(old_r+r);
    this->xt->conservativeResize(old_r+r);
}

Segment Cluster::head(){
    assert(this->x != nullptr);
    return this->x->segment(0, this->csize);
}

Segment Cluster::full(){
    assert(this->x != nullptr);
    return this->x->segment(0, this->rsize);
}

Segment Cluster::thead(){
    assert(this->xt != nullptr);
    return this->xt->segment(0, this->csize);
}

Segment Cluster::tfull(){
    assert(this->xt != nullptr);
    return this->xt->segment(0, this->rsize);
}

Eigen::VectorXd* Cluster::get_x(){return this->x;}

void Cluster::set_vector(const Eigen::VectorXd& b){
    assert(x != nullptr);
    assert(this->get_rstart() >=0);
    for (int i=0; i < this->rsize_org; ++i){
        (*this->get_x())[i] = b[this->get_rstart()+i];
    }
}

void Cluster::set_vector_x(const Eigen::VectorXd& b){
    assert(x != nullptr);
    assert(this->get_cstart()>=0);
    for (int i=0; i < this->csize_org; ++i){
        (*this->get_x())[i] = b[this->get_cstart()+i];
    }
}


void Cluster::tset_vector(const Eigen::VectorXd& b){
    assert(xt != nullptr);
    assert(this->get_cstart()>=0);
    for (int i=0; i < this->csize_org; ++i){
        (*xt)[i] = b[this->get_cstart()+i];
    }
}


void Cluster::extract_vector(Eigen::VectorXd& soln){
    assert(x != nullptr);
    for (int i=0; i < this->csize_org; ++i){
        soln[this->get_cstart()+i] = (*this->get_x())[i];
    }
}


void Cluster::textract_vector(){//Eigen::VectorXd& soln
    for (int i=0; i < this->rsize_org; ++i){
        (*x)[i] = (*xt)[i];
    }
}

void Cluster::textract_vector(Eigen::VectorXd& soln){
    assert(x != nullptr);
    for (int i=0; i < this->rsize_org; ++i){
        soln[this->get_rstart()+i] = (*this->xt)[i];
    }
}

void Cluster::reduce_x(Eigen::VectorXd& xc){
    std::lock_guard<std::mutex> lock(mutex_edgeIn); // NEEDED to avoid race condition
    assert(this->head().size() == xc.size());
    this->head() -= xc;
}

void Cluster::merge_x(){
    if (this->x_ex == nullptr) return; // nothing to merge
    int old_r = this->x->size();
    int n_ex = this->x_ex->size();
    this->x->conservativeResize(old_r+n_ex);
    this->x->segment(old_r, n_ex) = *(this->x_ex);
    delete this->x_ex;
}

/* Destructor */
Cluster::~Cluster(){
    delete x;
    delete tau;
    delete T;
}


