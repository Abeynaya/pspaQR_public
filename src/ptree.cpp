#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h> 
#include <iomanip> 
#include <random>
#include "tree.h"
#include "ptree.h"

using namespace std;
using namespace Eigen;
using namespace ttor;



void ParTree::set_nthreads(int n) { this->n_threads = n; };
void ParTree::set_verbose(int v) { this->verb = v; };
void ParTree::set_ttor_log(int l) { this->ttor_log = l; };

/* Helper function */
EdgeIt find_out_edge(Cluster* m, Cluster* n){
    auto found = find_if(m->edgesOut.begin(), m->edgesOut.end(), [&n](Edge* e_m){return e_m->n2 == n;});
    if (found == m->edgesOut.end()){
        found = find_if(m->edgesOutFillin.begin(), m->edgesOutFillin.end(), [&n](Edge* e_m){return e_m->n2 == n;});
        assert( found!= m->edgesOutFillin.end()); // Has to exist
    }
    return found;
}

bool ParTree::want_sparsify(Cluster* c) const{
    if (c->merge_lvl() < skip || c->merge_lvl() >= nlevels-2 || this->tol==0){
        return false;
    }
    if (c->lvl() > c->merge_lvl()){
        SepID left = c->get_id().l;
        SepID right = c->get_id().r;
        if (left.lvl <= c->merge_lvl() && right.lvl <= c->merge_lvl()) 
            return true;
    }
    return false;
}

/* Add an edge between c1 and c2
 such that A(c2, c1) block is non-zero */
Edge* ParTree::new_edgeOut(Cluster* c1, Cluster* c2){
    Edge* e;
    MatrixXd* A = new MatrixXd(c2->rows(), c1->cols());
    A->setZero();
    e = new Edge(c1, c2, A);
    c1->cnbrs.insert(c2);
    c1->add_edgeOut(e);

    bool is_spars_nbr = false;
    if (c1->lvl()>this->current_bottom && c2->lvl()>this->current_bottom && c1 != c2){
        c1->add_edge_spars_out(e);
        is_spars_nbr = true;
    }
    if(c1 != c2) c2->add_edgeIn(e,is_spars_nbr); // thread-safe using lock_guard
    return e; 
}

/* Add a fillin edge between c1 and c2
 such that A(c2, c1) block is non-zero */
Edge* ParTree::new_edgeOutFillin(Cluster* c1, Cluster* c2){
    Edge* e;
    MatrixXd* A = new MatrixXd(c2->rows(), c1->cols());
    A->setZero();
    e = new Edge(c1, c2, A);
    c1->cnbrs.insert(c2); // probably safe?
    c1->add_edgeOutFillin(e);

    bool is_spars_nbr = false;
    if (c1->lvl()>this->current_bottom && c2->lvl()>this->current_bottom && c1 != c2){
        c1->add_edge_spars_out(e);
        is_spars_nbr = true;
    }
    if (c1 != c2) c2->add_edgeInFillin(e,is_spars_nbr); // thread-safe using lock_guard
    return e; 
}

void ParTree::alloc_fillin(Cluster* c, Cluster* n){
    for (auto edge: c->edgesOut){
        Cluster* nbr_c = edge->n2; 
        auto found_out = find_if(n->edgesOut.begin(), n->edgesOut.end(), [&nbr_c](Edge* e){return (e->n2 == nbr_c );});
        auto found_fillin = n->edgesOutFillin.end();
        if (found_out == n->edgesOut.end()){
            found_fillin = find_if(n->edgesOutFillin.begin(), n->edgesOutFillin.end(), [&nbr_c](Edge* e) {return (e->n2 == nbr_c);});
            if (found_fillin == n->edgesOutFillin.end()){ // Cannot find the edge
                Edge* ecn = new_edgeOutFillin(n, nbr_c);
                ecn->interior_deps++;
            }
            else {
                (*found_fillin)->interior_deps++;
            }
        }
        else {
            (*found_out)->interior_deps++;
        }
    }
    return;
}

/* Elimination */
void ParTree::geqrt_cluster(Cluster* c){
    if (c->cols()==0){
        c->set_eliminated();
        cout << "cluster has zero cols" << endl;
        exit(1);
    }

    // Combine edgesIn and edgesInFillin -- just to be consistent
    c->combine_edgesIn(); // No race condition

    // Create matrices to store triangular factor T for all edgesOut
    for (auto e: c->edgesOut){
        c->create_T(e->n2->get_order());
    }

    // Get self->edge
    int nrows = c->rows();
    int ncols = c->cols();
    MatrixXd T = MatrixXd::Zero(ncols, ncols);

    int info = LAPACKE_dgeqrt(LAPACK_COL_MAJOR, nrows, ncols, ncols, c->self_edge()->A21->data(), nrows, T.data(), ncols);
    assert(info == 0);
    c->set_T(c->get_order(), T);
    c->set_eliminated();

    return;
}

void ParTree::tsqrt_edge(Edge* e){
    Cluster* c = e->n1;
    Cluster* n = e->n2;
    assert(c->is_eliminated());
    int nrows = e->A21->rows();
    int ncols = e->A21->cols();
    int nb=min(32,c->cols());
    MatrixXd T = MatrixXd::Zero(nb, ncols);
    int info = LAPACKE_dtpqrt(LAPACK_COL_MAJOR,nrows, ncols, 0, nb,  // using big nb value causes error -- MKl bug?
                              c->self_edge()->A21->data(), c->rows(), 
                              e->A21->data(), nrows,
                              T.data(), nb);
    assert(info == 0);
    c->set_T(n->get_order(), T);
    return;
}

void ParTree::larfb_edge(Edge* e){ // A12 block
    Cluster* n = e->n1;
    Cluster* c = e->n2;
    assert(c->is_eliminated());

    int nrows = e->A21->rows();
    int ncols = e->A21->cols();

    larfb(c->self_edge()->A21, c->get_T(c->get_order()), e->A21);
    return;
}

void ParTree::ssrfb_edges(Edge* cn, Edge* cm, Edge* mn){ // c->n edge contains Q and m->n contains matrix to apply Q on
    Cluster* c = cn->n1;
    Cluster* n = cn->n2;
    Cluster* m = mn->n1; 
    assert(mn->n2 == n);
    assert(c->is_eliminated());

    int nrows = mn->A21->rows();
    int ncols = mn->A21->cols();

    assert(cn->A21->rows() ==  nrows);
    assert(cm->A21->rows() == c->rows());
    assert(mn->A21->rows() == nrows);
    
    int nb=min(32,c->cols());
    // use directly from mkl_lapack.h -- routine from mkl_lapacke.h doesn't work
    char side = 'L';
    char trans = 'T';
    int k = c->cols();
    int lda = c->rows();
    int l=0;
    VectorXd work = VectorXd::Zero(ncols*nb);
    int info=-1;
    dtpmqrt_(&side, &trans, &nrows, &ncols, &k, &l, &nb, 
                                cn->A21->data(), &nrows, 
                                c->get_T(n->get_order())->data(), &nb,
                                cm->A21->data(), &lda,
                                mn->A21->data(), &nrows,
                                work.data(), &info);
    assert(info == 0);
    return;
}

/* Scaling */
void ParTree::scale_cluster(Cluster* c){
    Edge* e = c->self_edge();
    assert(e != nullptr);
    MatrixXd Q = *(e->A21);
    MatrixXd T = MatrixXd::Zero(c->cols(), c->cols());

    // VectorXd t = VectorXd::Zero(c->cols());
    // geqrf(&Q, &t);
    // larft(&Q, &t, &T);

    int ncols = Q.cols();
    int nrows = Q.rows();
    int info = LAPACKE_dgeqrt(LAPACK_COL_MAJOR, nrows, ncols, ncols, Q.data(), nrows, T.data(), ncols);
    assert(info == 0);

    // Setting this is important
    (*e->A21).topRows(c->cols()) = MatrixXd::Identity(c->cols(), c->cols());
    (*e->A21).bottomRows(c->rows()-c->cols()) = MatrixXd::Zero(c->rows()-c->cols(),c->cols());

    c->set_Qs(Q);
    c->set_Ts(T);
    // c->set_taus(t);
}

/* Merge */
void ParTree::compute_new_edges(Cluster* snew){
    set<Cluster*> edges_merged;
    for (auto sold: snew->children){
        sold->combine_edgesOut(); // IMPORTANT -- safe since each snew has unique children
        for (auto eold : sold->edgesOut){
            auto n = eold->n2; 
            if (!(n->is_eliminated())){
                assert(n->get_parent() != nullptr);
                snew->cnbrs.insert(n->get_parent());
            }
        }
    }
    
    // Allocate memory and create new edges
    for (auto n: snew->cnbrs){
        Edge* e = new_edgeOut(snew, n);
        if (snew==n) snew->add_self_edge(e);
        else if (snew->lvl() == snew->merge_lvl()) n->col_interior_deps++;
    }

    // Sort edges
    snew->sort_edgesOut();

    // Fill edges, delete previous edges
    for (auto sold: snew->children){
        for (auto eold: sold->edgesOut){
            auto nold = eold->n2;

            if (!(nold->is_eliminated())){
                auto nnew = nold->get_parent();
                auto found = find_if(snew->edgesOut.begin(), snew->edgesOut.end(), [&nnew](Edge* e){return e->n2 == nnew;});
                assert(found != snew->edgesOut.end());  // Must have the edge

                assert(eold->A21 != nullptr);
                if (eold->A21->rows() != nold->rows()) cout << eold->n1->get_id() << " " << eold->n2->get_id() <<" " << eold->A21->rows() << " " << nold->rows() << endl;
                assert(eold->A21->rows() == nold->rows());
                assert(eold->A21->cols() == sold->cols());

                (*found)->A21->block(nold->rposparent, sold->cposparent, nold->rows(), sold->cols()) = *(eold->A21);
                // delete eold; // just free memory
                delete eold->A21; // free memory
                eold = nullptr; // set it to nullptr but it is not removed from the list
            }
            
        }
    }
}

void ParTree::trsm_edge(Edge* e){ // Applied on edgesOut of the cluster being scaled
    Cluster* c = e->n1; 
    MatrixXd R = c->get_Qs()->topRows(c->cols()).triangularView<Upper>();

    trsm_right(&R, e->A21, CblasUpper, CblasNoTrans, CblasNonUnit);
}

/* Sparsification */
void ParTree::sparsify_rrqr_only(Cluster* c){
    // bool sparsified = true;
    // if (c->cols() == 0) sparsified=false;
    // else {
    if (!want_sparsify(c)) return;
    assert(c->cols() != 0);
    vector<MatrixXd*> Apm(c->edgesInNbrSparsification.size()); // row nbrs
    vector<MatrixXd*> Amp(c->edgesOutNbrSparsification.size()); // col nbrs
    Edge* e_self = c->self_edge();
    assert(e_self != nullptr);

    int spm=0;
    int smp=0;
    int count=0;
    for (auto e: c->edgesInNbrSparsification){
        Apm[count]=e->A21;
        spm += e->A21->cols();
        count++;
    }
    count=0;
    for (auto e: c->edgesOutNbrSparsification){
        Amp[count] = e->A21;
        smp += e->A21->rows();
        count++;
    }

    MatrixXd C = MatrixXd::Zero(c->cols(), spm+smp); // Block to be compressed
    MatrixXd Apmc = Hconcatenate(Apm);

    C.leftCols(smp) = Vconcatenate(Amp).transpose();
    C.middleCols(smp, spm) = Apmc.topRows(c->cols());

    int rank = 0;
    VectorXi jpvt = VectorXi::Zero(C.cols());
    VectorXd t = VectorXd(min(C.rows(), C.cols()));
    VectorXd rii;
   
    #ifdef USE_MKL
        /* Rank revealing QR on C with rrqr function */ 
        int bksize = min(C.cols(), max((long)3,(long)(0.05 * C.rows())));
        laqps(&C, &jpvt, &t, this->tol, bksize, rank);
        rii = C.topLeftCorner(rank, rank).diagonal();

    #else
        geqp3(&C, &jpvt, &t);
        rii = C.diagonal();
    #endif

    rank = choose_rank(rii, this->tol);
    assert(rank <= min(C.rows(), C.cols()));

    /* Sparsification */
    MatrixXd v = MatrixXd(C.rows(), rank);
    VectorXd h = VectorXd(rank);
    MatrixXd T = MatrixXd::Zero(rank, rank);

    v = C.leftCols(rank);
    h = t.topRows(rank);

    // do larft 
    larft(&v, &h, &T);
    c->set_Q_sp(v);
    c->set_tau_sp(h); 
    c->set_T_sp(T);

    // Change size of pivot block
    *(e_self->A21) = MatrixXd::Identity(rank, rank);
    c->set_rank(rank);
    c->reset_size(c->get_rank(), c->get_rank());

}

int ParTree::factorize(){
	auto fstart = systime();
    assert(this->scale==1);
    cout << "Lvl " << "Fillin " <<  "Sprsfy " << "Merge " << "Total " <<  "  s_top " << "Threads(b,t) "<< endl;
    for (this->ilvl=0; this->ilvl < nlevels; ++ilvl){
        auto lvl0 = systime();
        timeval vstart, vend, estart, eend, mstart, mend;

        int blas_threads;
        int ttor_threads;

        // Find fill-in and allocate memory
        {
            blas_threads=1;
            ttor_threads=n_threads;

            {
                auto t0 = wctime();
                #ifdef USE_MKL
                std::string kind = "MKL";
                mkl_set_num_threads(blas_threads);
                if (blas_threads==1) mkl_set_dynamic(0);
                #else
                std::string kind = "OpenBLAS";
                openblas_set_num_threads(blas_threads);
                #endif
                auto t1 = wctime();
                if(this->verb) printf("  n_threads %d, (kind %s, time %3.2e s.) => ttor threads %d, blas threads %d\n", n_threads, kind.c_str(), elapsed(t0, t1), ttor_threads, blas_threads);
                assert(ttor_threads * blas_threads <= n_threads);
            }

            // Threadpool
            Threadpool_shared tp(ttor_threads, verb, "fillin_"+ to_string(ilvl)+"_");
            Taskflow<Cluster*> get_sparsity_tf(&tp, verb);
            Taskflow<pCluster2> fillin_tf(&tp, verb);

            Logger log(1000); 
            if (this->ttor_log){
                tp.set_logger(&log);
            }

            get_sparsity_tf.set_mapping([&] (Cluster* c){
                    return c->get_order()%ttor_threads; 
                })
                .set_indegree([](Cluster*){
                    return 1;
                })
                .set_task([&] (Cluster* c) {
                    this->get_sparsity(c);
                })
                .set_fulfill([&] (Cluster* c){
                    for (auto n: c->rsparsity){
                        fillin_tf.fulfill_promise({c,n});
                    }
                })
                .set_name([](Cluster* c) {
                    return "get_sparsity_" + to_string(c->get_order());
                })
                .set_priority([&](Cluster*) {
                    return 6;
                });

            fillin_tf.set_mapping([&] (pCluster2 cn){
                    return cn[1]->get_order()%ttor_threads; // Important to avoid race conditions 
                })
                .set_binding([] (pCluster2) {
                    return true;
                })
                .set_indegree([](pCluster2 cn){
                    return 1;  // get_sparsity on c
                })
                .set_task([&] (pCluster2 cn) {
                    alloc_fillin(cn[0], cn[1]);
                })
                .set_name([](pCluster2 cn) {
                    return "fillin_" + to_string(cn[0]->get_order()) + "_" + to_string(cn[1]->get_order());
                })
                .set_priority([&](pCluster2) {
                    return 7;
                });

            // Get rsparsity for the clusters to be eliminated next
            vstart = systime();
            {
                for (auto self: this->interiors_current()){
                    get_sparsity_tf.fulfill_promise(self);
                }
            }
            tp.join();
            vend = systime();

            auto log0 = systime();
            if (this->ttor_log){
                std::ofstream logfile;
                std::string filename = "./logs/fillin_" + to_string(this->ilvl) + ".log." + to_string(0);
                if(this->verb) printf("  Level %d logger saved to %s\n", this->ilvl, filename.c_str());
                logfile.open(filename);
                logfile << log;
                logfile.close();
            }
            auto log1 = systime();

        }
        // Eliminate, Scale and Sparsify
        {
            // Count the number of sparsification tasks
            const int real_tasks = this->interfaces_current().size();
            const int tasks = std::max(real_tasks/4,1); // Make it more aggressive. Gives more threads to BLAS towards the end (it's very unbalanced anyway).
                                                        // Otherwise we may divide by 0
            assert(tasks > 0);

            blas_threads = 1;
            /*
            if(tasks > this->n_threads) {
                blas_threads = 1;
            } else {
                blas_threads = this->n_threads / tasks;
            }
            */
            assert(blas_threads > 0);
            assert(blas_threads <= n_threads);
            ttor_threads = n_threads / blas_threads;
            assert(ttor_threads > 0);
            assert(ttor_threads <= n_threads);

            // Threadpool
            Threadpool_shared tp(ttor_threads, verb, "elmn_" + to_string(ilvl)+"_");
            /* Elimination */
            Taskflow<Cluster*> geqrt_tf(&tp, verb);
            Taskflow<EdgeIt> larfb_tf(&tp, verb);
            Taskflow<EdgeIt3> ssrfb_tf(&tp, verb);
            Taskflow<EdgeIt> tsqrt_tf(&tp, verb);
            /* Scale */
            Taskflow<Cluster*> scale_geqrt_tf(&tp, verb);
            Taskflow<Edge*> scale_larfb_trsm_tf(&tp, verb);
            /* Sparsify */
            Taskflow<Cluster*> sparsify_rrqr_tf(&tp, verb);
            Taskflow<Edge*> sparsify_larfb_tf(&tp, verb);

            {
                auto t0 = wctime();
                #ifdef USE_MKL
                std::string kind = "MKL";
                mkl_set_num_threads(blas_threads);
                if (blas_threads==1) mkl_set_dynamic(0);
                #else
                std::string kind = "OpenBLAS";
                openblas_set_num_threads(blas_threads);
                #endif
                auto t1 = wctime();
                if(this->verb) printf("  n_threads %d, spars tasks %d (real %d, kind %s, time %3.2e s.) => ttor threads %d, blas threads %d\n", n_threads, tasks, real_tasks, kind.c_str(), elapsed(t0, t1), ttor_threads, blas_threads);
                assert(ttor_threads * blas_threads <= n_threads);
            }

            Logger log(10000000); 
            if (this->ttor_log){
                // log = Logger(10000000);
                tp.set_logger(&log);
            }

            geqrt_tf.set_mapping([&] (Cluster* c){
                    return (c->get_order()+1)%ttor_threads; 
                })
                .set_indegree([](Cluster*){
                    return 1;
                })
                .set_task([&] (Cluster* c) {
                    geqrt_cluster(c);
                })
                .set_fulfill([&] (Cluster* c){
                    // All larfb
                    for (EdgeIt it=c->edgesIn.begin(); it!=c->edgesIn.end(); ++it){ // All the edges that need to be updated
                        larfb_tf.fulfill_promise(it);
                    }
                    // First tsqrt
                    auto c_edge_it = c->edgesOut.begin();
                    assert((*c_edge_it)->n2 == c); // first edgeout is c itself when c is at leaf level
                    c_edge_it++;
                    if (c_edge_it != c->edgesOut.end()){
                        tsqrt_tf.fulfill_promise(c_edge_it);
                    }
                })
                .set_name([](Cluster* c) {
                    return "geqrt_" + to_string(c->get_order());
                })
                .set_priority([&](Cluster*) {
                    return 6;
                });

            larfb_tf.set_mapping([&] (EdgeIt eit){
                    Edge* e = *eit;
                    return (e->n1->get_order() + e->n2->get_order())%ttor_threads;  
                })
                .set_indegree([](EdgeIt){ // e->A21 = A_cn
                    return 1; // geqrt on e->n2
                })
                .set_task([&] (EdgeIt eit) {
                    larfb_edge(*eit);
                })
                .set_fulfill([&] (EdgeIt eit){
                    // EdgeIt eit_next = eit;
                    Edge* e = *eit;
                    Cluster* c = e->n2;
                    Cluster* m = e->n1;
                    // First ssrfb
                    auto c_edge_it = c->edgesOut.begin();
                    assert((*c_edge_it)->n2 == c); // first edgeout is c itself when c is at leaf level
                    c_edge_it++;

                    if (c_edge_it != c->edgesOut.end()){
                        Edge* eout = *c_edge_it;
                        Cluster*  n = eout->n2;
                        auto found = find_out_edge(m,n);
                        ssrfb_tf.fulfill_promise({c_edge_it, eit, found}); // cn, cm, mn
                    }
                })
                .set_name([](EdgeIt eit) {
                    Edge* e = *eit;
                    return "larfb_" + to_string(e->n1->get_order())+ "_" + to_string(e->n2->get_order());
                })
                .set_priority([&](EdgeIt) {
                    return 4;
                });

            ssrfb_tf.set_mapping([&] (EdgeIt3 it){ // needs binding
                    Edge* e_mn = *it[2];
                    return (e_mn->n1->get_order() + e_mn->n2->get_order())%ttor_threads;  
                })
                .set_binding([] (EdgeIt3){ // IMPORTANT to avoid race condition
                    return true;
                })
                .set_indegree([](EdgeIt3){ 
                    return 2;  // tsqrt and larfb or tsqrt and ssrfb
                })
                .set_task([&] (EdgeIt3 it) {
                    ssrfb_edges(*it[0], *it[1], *it[2]);
                })
                .set_fulfill([&] (EdgeIt3 it){
                    auto c_edge_it = it[0]; // cn
                    Cluster* c = (*c_edge_it)->n1;

                    // First fulfill scale 
                    Edge* edge_mn = *it[2];
                    if (this->ilvl >= skip && this->ilvl < nlevels-2  && this->tol != 0){
                        if (edge_mn->n2 == edge_mn->n1) scale_geqrt_tf.fulfill_promise(edge_mn->n1); // Scale GEQRF
                        else scale_larfb_trsm_tf.fulfill_promise(edge_mn);
                    }

                    // Next ssrfb
                    c_edge_it++;
                    Cluster* m = (*it[1])->n1;
                    if (c_edge_it != c->edgesOut.end()){
                        Edge* eout = *c_edge_it;
                        Cluster*  n = eout->n2;
                        auto found = find_out_edge(m,n);
                        ssrfb_tf.fulfill_promise({c_edge_it, it[1], found}); // cn, cm, mn
                    }

                })
                .set_name([](EdgeIt3 it) {
                    Edge* e_cn = *it[0];
                    Edge* e_cm = *it[1];

                    return "ssrfb_" + to_string(e_cn->n1->get_order())+ "_" + to_string(e_cn->n2->get_order())
                        +"_" + to_string(e_cm->n1->get_order());
                })
                .set_priority([&](EdgeIt3) {
                    return 3;
                });

            // should I bind this to get better memory access times?
            tsqrt_tf.set_mapping([&] (EdgeIt eit){
                    Edge* e = *eit;
                    return (e->n1->get_order() + 1)%ttor_threads;  
                })
                .set_indegree([](EdgeIt){ // e->A21 = A_cn
                    return 1; // geqrt on e->n1
                })
                .set_task([&] (EdgeIt eit) {
                    tsqrt_edge(*eit);
                })
                .set_fulfill([&] (EdgeIt eit){
                    auto eit_next = eit;
                    eit_next++;
                    Cluster* c = (*eit)->n1;
                    Cluster* n = (*eit)->n2;
                    // Next tsqrt
                    if (eit_next != c->edgesOut.end()){
                        tsqrt_tf.fulfill_promise(eit_next);
                    }
                    // All ssrfb in its row
                    for (EdgeIt it=c->edgesIn.begin(); it!=c->edgesIn.end(); ++it){ // All the edges that need to be updated
                        Cluster* m = (*it)->n1;
                        auto found = find_out_edge(m,n);
                        ssrfb_tf.fulfill_promise({eit, it, found}); // cn, cm, mn
                    }
                })
                .set_name([](EdgeIt eit) {
                    Edge* e = *eit;
                    return "tsqrt_" + to_string(e->n1->get_order())+ "_" + to_string(e->n2->get_order());
                })
                .set_priority([&](EdgeIt) {
                    return 6;
                });

            /* Scaling */
            scale_geqrt_tf.set_mapping([&] (Cluster* c){
                    return c->get_order()%ttor_threads; 
                })
                .set_indegree([&](Cluster* c){
                    if (c->self_edge()->interior_deps==0) return (unsigned long)1;
                    return c->self_edge()->interior_deps;
                })
                .set_task([&] (Cluster* c) {
                    if (verb) cout << "scale_" << c->get_id() << endl;
                    if(want_sparsify(c)) scale_cluster(c);
                })
                .set_fulfill([&] (Cluster* c){
                    // Fulfill these even for clusters that are not sparsified -- important
                    // Apply ormqr to rows and cols
                    for (auto ein: c->edgesInNbrSparsification){
                        scale_larfb_trsm_tf.fulfill_promise(ein);
                    }
                    for (auto eout: c->edgesOutNbrSparsification){
                        scale_larfb_trsm_tf.fulfill_promise(eout);
                    }
                    // Sparsify cluster
                    sparsify_rrqr_tf.fulfill_promise(c);
                })
                .set_name([](Cluster* c) {
                    return "scale_geqrf_" + to_string(c->get_order());
                })
                .set_priority([&](Cluster*) {
                    return 5;
                });

            /* Apply Q_c^T A_cn and R_n\A_cn 
             * Q_c, R_n from scaling */
            scale_larfb_trsm_tf.set_mapping([&] (Edge* e){
                    return (e->n1->get_order() + e->n2->get_order())%ttor_threads;  
                })
                .set_indegree([](Edge* e){ // e->A21 = A_cn
                    return 2+ e->interior_deps;
                })
                .set_task([&] (Edge* e) {
                    if (verb) cout << "scale_apply_" << e->n1->get_id() << " " << e->n2->get_id() << endl;
                    Cluster* c = e->n2;
                    Cluster* n = e->n1; 
                    if (want_sparsify(c)) larfb(c->get_Qs(), c->get_Ts(), e->A21);
                    if (want_sparsify(n)) trsm_edge(e);
                })
                .set_fulfill([&] (Edge* e){
                    // Sparsify cluster
                    sparsify_rrqr_tf.fulfill_promise(e->n1);
                    sparsify_rrqr_tf.fulfill_promise(e->n2);
                })
                .set_name([](Edge* e) {
                    return "scale_apply_" + to_string(e->n1->get_order())+ "_" + to_string(e->n2->get_order());
                })
                .set_priority([&](Edge*) {
                    return 4;
                });

            /* Sparsification */
            sparsify_rrqr_tf.set_mapping([&] (Cluster* c){
                    return c->get_order()%ttor_threads; 
                })
                .set_indegree([](Cluster* c){
                    return c->edgesOutNbrSparsification.size()  // trsm from scaling on cols
                            + c->edgesInNbrSparsification.size()  // larfb from scaling on rows
                            + 1;  // scaling on itself
                })
                .set_task([&] (Cluster* c) {
                    if(verb) cout << "sparsify_rrqr_tf_" << c->get_id() << endl;
                    if(want_sparsify(c)) sparsify_rrqr_only(c);
                })
                .set_fulfill([&] (Cluster* c){
                    // Apply ormqr to rows and cols
                    for (auto ein: c->edgesInNbrSparsification){
                        sparsify_larfb_tf.fulfill_promise(ein); // Q_c^T ein->A21
                    }
                    for (auto eout: c->edgesOutNbrSparsification){
                        sparsify_larfb_tf.fulfill_promise(eout); // eout->A21* Q_c
                    } 
                })
                .set_name([](Cluster* c) {
                    return "sparsify_rrqr_" + to_string(c->get_order());
                })
                .set_priority([&](Cluster*) {
                    return 4;
                });

            /* Apply Q_c^T A_cn and A_cn Q_n 
             * Q_c, Q_n from sparsification */
            sparsify_larfb_tf.set_mapping([&] (Edge* e){
                    return (e->n1->get_order()+e->n2->get_order())%ttor_threads;  
                })
                .set_indegree([](Edge*){
                    return 2; // the sparsify on clusters c and n
                })
                .set_task([&] (Edge* e) {
                    if (verb) cout << "sparsify_apply_" << e->n1->get_id() << " " << e->n2->get_id() << endl;
                    Cluster* c = e->n2; 
                    Cluster* n = e->n1; 
                    MatrixXd Aold = *e->A21;
                    // 1. A_cn <- Q_c^T A_cn
                    if (want_sparsify(c)) larfb(c->get_Q_sp(), c->get_T_sp(), &Aold);
                    // 2. A_cn <- A_cn Q_n
                    if (want_sparsify(n)) larfb_notrans_right(n->get_Q_sp(), n->get_T_sp(), &Aold);
                    *e->A21 = Aold.block(0,0,c->get_rank(),n->get_rank());
                })
                .set_name([](Edge* e) {
                    return "sparsify_apply_" + to_string(e->n1->get_order())+ "_" + to_string(e->n2->get_order());
                })
                .set_priority([&](Edge*) {
                    return 3;
                });

            // Eliminate interiors
            estart = systime();
            {
                for (auto self: this->interiors_current()){ 
                    geqrt_tf.fulfill_promise(self);
                }

                for (auto self: this->interfaces_current()){ 
                    if (this->ilvl < nlevels-2  && this->ilvl >= skip && this->tol != 0){
                        if (self->self_edge()->interior_deps == 0){
                            scale_geqrt_tf.fulfill_promise(self);
                        }
                    }
                }
            }
            tp.join();
            eend = systime();

            auto log0 = systime();
            if (this->ttor_log){
                std::ofstream logfile;
                std::string filename = "./logs/spars_" + to_string(this->ilvl) + ".log." + to_string(0);
                if(this->verb) printf("  Level %d logger saved to %s\n", this->ilvl, filename.c_str());
                logfile.open(filename);
                logfile << log;
                logfile.close();
            }
            auto log1 = systime();
        }

        // Merge
        mstart = systime();
        if (this->ilvl < nlevels-1)
        {
            this->current_bottom++;
            // Update sizes -- sequential because we need a synch point after this and this is cheap
            for (auto snew: this->bottom_current()){
                int rsize = 0;
                int csize = 0;
                for (auto sold: snew->children){
                    sold->rposparent = rsize;
                    sold->cposparent = csize;
                    rsize += sold->rows();
                    csize += sold->cols();
                    for (auto d2c: sold->dist2connxs) if(!d2c->is_eliminated()) snew->dist2connxs.insert(d2c->get_parent());
                }
                snew->set_size(rsize, csize); 
                snew->set_org(rsize, csize);
            }

            // Update edges
            Threadpool_shared tp(n_threads, verb, "merge_"+ to_string(ilvl)+"_");
            Taskflow<Cluster*> update_edges_tf(&tp, verb);
            
            Logger log(1000); 
            if (this->ttor_log){
                tp.set_logger(&log);
            }

            /* Compute new edges */
            update_edges_tf.set_mapping([&] (Cluster* c){
                    return (c->get_order()+1)%n_threads; 
                })
                .set_indegree([](Cluster*){
                    return 1;
                })
                .set_task([&] (Cluster* c) {
                    compute_new_edges(c);
                })
                .set_name([](Cluster* c) {
                    return "update_edges_" + to_string(c->get_order());
                })
                .set_priority([&](Cluster*) {
                    return 5;
                });

            for (auto snew: this->bottom_current()){
                update_edges_tf.fulfill_promise(snew);
            }
            tp.join();

            auto log0 = systime();
            if (this->ttor_log){
                std::ofstream logfile;
                std::string filename = "./logs/merge_" + to_string(this->ilvl) + ".log." + to_string(0);
                if(this->verb) printf("  Level %d logger saved to %s\n", this->ilvl, filename.c_str());
                logfile.open(filename);
                logfile << log;
                logfile.close();
            }
            auto log1 = systime();

        }
        mend = systime();
        auto lvl1 = systime();

        cout << ilvl << "    " ;  
        cout << fixed << setprecision(3)  
             <<  elapsed(vstart,vend) << "   "
             <<  elapsed(estart,eend) << "   "
        //      << "shift: " <<  elapsed(shstart,shend) << "  "
             // << "scale: " << elapsed(scstart,scend) << "  "
             // << "sprsfy: " << elapsed(spstart,spend) << "  "
             <<  elapsed(mstart,mend) << "  " 
             <<  elapsed(lvl0,lvl1) << "  ("
             <<  get<0>(topsize()) << ", " << get<1>(topsize()) << ")   (" 
             <<  blas_threads << ", " << ttor_threads << ")  " 
        //      << "a.r top_sep: " << (double)get<0>(topsize())/(double)get<1>(topsize()) 
             << endl;

        
    }
    auto fend = systime();
    cout << "Tolerance set: " << scientific << this->tol << endl;
    cout << "Time to factorize:  " << elapsed(fstart,fend) << endl;

	return 0;
}


/**
 * Solve
 **/
void ParTree::QR_fwd(Cluster* c) const{
    // geqrt on self_edge->A21
    MatrixXd* cc = c->self_edge()->A21;
    int cx_size = cc->rows();
    Segment xc = c->get_x()->segment(0,cx_size);
    larfb(cc, c->get_T(c->get_order()), &xc);
    

    // tsqrt on all other c->edgesOut
    for (auto e: c->edgesOut){
        Cluster* n = e->n2;
        if (n != c){
            MatrixXd* nc = e->A21;
            int nrows = nc->rows(); // will not be affected even if e->n2->rows() change b/c of sparsification
            int ncols = 1;
            int k = cc->cols();
            int nb = min(32, k);
            Segment xn = n->get_x()->segment(0,nrows);

            // use directly from mkl_lapack.h -- routine from mkl_lapacke.h doesn't work correctly
            char side = 'L';
            char trans = 'T';
            int l=0;
            VectorXd work = VectorXd::Zero(ncols*nb);
            int info=-1;
            dtpmqrt_(&side, &trans, &nrows, &ncols, &k, &l, &nb, 
                                        nc->data(), &nrows, 
                                        c->get_T(n->get_order())->data(), &nb,
                                        xc.data(), &cx_size,
                                        xn.data(), &nrows,
                                        work.data(), &info);
            assert(info == 0);
        }
    }
}

void ParTree::QR_tsqrt_fwd(Edge* e) const{
    Cluster* c = e->n1;
    Cluster* n = e->n2;
    MatrixXd* A_nc = e->A21;
    int nrows = A_nc->rows(); // will not be affected even if e->n2->rows() change b/c of sparsification
    int ncols = 1;
    int k = c->cols();
    int cx_size = c->rows();
    int nb = min(32, k);
    Segment xc = c->get_x()->segment(0,cx_size);
    Segment xn = n->get_x()->segment(0,nrows);

    // use directly from mkl_lapack.h -- routine from mkl_lapacke.h doesn't work correctly
    char side = 'L';
    char trans = 'T';
    int l=0;
    VectorXd work = VectorXd::Zero(ncols*nb);
    int info=-1;
    dtpmqrt_(&side, &trans, &nrows, &ncols, &k, &l, &nb, 
                                A_nc->data(), &nrows, 
                                c->get_T(n->get_order())->data(), &nb,
                                xc.data(), &cx_size,
                                xn.data(), &nrows,
                                work.data(), &info);
    assert(info == 0);
}

void ParTree::QR_bwd(Cluster* c) const{
    int i=0;
    Segment xs = c->get_x()->segment(0, c->cols());
    for (auto e: c->edgesIn){
        int s = e->A21->cols();
        xs -= (e->A21->topRows(c->cols()))*(e->n1->get_x()->segment(0, s)); 
        ++i;    
    }
    MatrixXd R = c->self_edge()->A21->topRows(xs.size()); // correct when using geqrt
    // MatrixXd R = c->get_Q()->topRows(xs.size()); // used this in the earlier implementation
    trsv(&R, &xs, CblasUpper, CblasNoTrans, CblasNonUnit); 
}

void ParTree::scaleD_fwd(Cluster* c) const{
    Segment xs = c->get_x()->segment(0,c->original_rows());
    larfb(c->get_Qs(), c->get_Ts(), &xs);
}

void ParTree::scaleD_bwd(Cluster* c) const{
    Segment xs = c->get_x()->segment(0,c->original_cols());
    MatrixXd R = c->get_Qs()->topRows(xs.size());
    trsv(&R, &xs, CblasUpper, CblasNoTrans, CblasNonUnit);
}

void ParTree::orthogonalD_fwd(Cluster* c) const{
    Segment xs = c->get_x()->segment(0,c->original_cols());
    ormqr_trans(c->get_Q_sp(), c->get_tau_sp(), &xs);
}

void ParTree::orthogonalD_bwd(Cluster* c) const{
    Segment xs = c->get_x()->segment(0,c->original_cols());
    ormqr_notrans(c->get_Q_sp(), c->get_tau_sp(), &xs);
}

void ParTree::merge_fwd(Cluster* parent) const{
    int k=0;
    for (auto c: parent->children){
        for (int i=0; i < c->rows(); ++i){
            (*parent->get_x())[k] = (*c->get_x())[i];
            ++k;
        }
    }
}

void ParTree::merge_bwd(Cluster* parent) const{ 
    int k=0;
    for (auto c: parent->children){
        for (int i=0; i < c->cols(); ++i){
            (*c->get_x())[i] = (*parent->get_x())[k];
            ++k;
        }
    }
}


// /* For linear systems */
void ParTree::solve(VectorXd b, VectorXd& x) const{
    // Permute the rhs according to this->rperm
    b = this->rperm.asPermutation().transpose()*b;

    // Set solution -- keep it sequential 
    for (auto cluster: bottom_original()){
        cluster->set_vector(b);
    }

    // Fwd
    {
        const int ttor_threads = n_threads;
        #ifdef USE_MKL
            mkl_set_num_threads(1);
            mkl_set_dynamic(0); // no dynamic multi-threading
        #else
            openblas_set_num_threads(1);
        #endif

        // Threadpool
        Threadpool_shared tp(ttor_threads, verb, "solve_");
        /* Elimination */
        Taskflow<Cluster*> fwd_geqrt(&tp, verb);
        Taskflow<EdgeIt> fwd_tsqrt(&tp, verb);

        /* Scale and Sparsify */
        Taskflow<Cluster*> fwd_spars(&tp, verb);
        /* Merge */
        Taskflow<Cluster*> fwd_merge(&tp, verb);

        fwd_geqrt.set_mapping([&] (Cluster* c){
                return (c->get_order()+1)%ttor_threads; 
            })
            .set_indegree([](Cluster*){
                return 1;
            })
            .set_task([&] (Cluster* c) {
                MatrixXd* cc = c->self_edge()->A21;
                int cx_size = cc->rows();
                Segment xc = c->get_x()->segment(0,cx_size);
                larfb(cc, c->get_T(c->get_order()), &xc);
            })
            .set_fulfill([&] (Cluster* c){
                // First tsqrt
                auto c_edge_it = c->edgesOut.begin();
                assert((*c_edge_it)->n2 == c); // first edgeout is c itself when c is at leaf level
                c_edge_it++;
                if (c_edge_it != c->edgesOut.end()){
                    fwd_tsqrt.fulfill_promise(c_edge_it);
                }
            })
            .set_name([](Cluster* c) {
                return "fwd_geqrt_" + to_string(c->get_order());
            })
            .set_priority([&](Cluster*) {
                return 6;
            });

        fwd_tsqrt.set_mapping([&] (EdgeIt eit){
                Edge* e = *eit;
                return e->n2->get_order() % ttor_threads;  // Important to avoid race conditions
            })
            .set_indegree([](EdgeIt eit){ // e->A21 = A_cn
                Edge* e = *eit;
                return (e->n1->merge_lvl()==0 ? 1:2); // geqrt on e->n1 and merge
            })
            .set_binding([] (EdgeIt){
                return true; 
            })
            .set_task([&] (EdgeIt eit) {
                QR_tsqrt_fwd(*eit);
            })
            .set_fulfill([&] (EdgeIt eit){
                auto eit_next = eit;
                eit_next++;
                Cluster* c = (*eit)->n1;
                Cluster* n = (*eit)->n2;
                // Next tsqrt
                if (eit_next != c->edgesOut.end()){
                    fwd_tsqrt.fulfill_promise(eit_next);
                }
                // Sparsification or merge
                fwd_spars.fulfill_promise(n); // always
            })
            .set_name([](EdgeIt eit) {
                Edge* e = *eit;
                return "fwd_tsqrt_" + to_string(e->n1->get_order())+ "_" + to_string(e->n2->get_order());
            })
            .set_priority([&](EdgeIt) {
                return 6;
            });

        fwd_spars.set_mapping([&] (Cluster* c){
                return c->get_order()%ttor_threads; 
            })
            .set_indegree([&](Cluster* c){
                if (c->col_interior_deps==0) return (unsigned long int)1;
                return c->col_interior_deps; 
            })
            .set_task([&] (Cluster* c) {
                // Apply Q from scaling and spars if any
                if (want_sparsify(c)){
                    scaleD_fwd(c); // FWD scaling
                    orthogonalD_fwd(c); // FWD sparsification
                }
            })
            .set_fulfill([&] (Cluster* c){
                if (c->merge_lvl() < nlevels-1) fwd_merge.fulfill_promise(c->get_parent());
            })
            .set_name([](Cluster* c) {
                return "fwd_spars_" + to_string(c->get_order());
            })
            .set_priority([&](Cluster*) {
                return 5;
            });

        fwd_merge.set_mapping([&] (Cluster* c){
                return c->get_order()%ttor_threads; 
            })
            .set_indegree([&](Cluster* c){
                return c->children.size();
            })
            .set_task([&] (Cluster* c) {
                merge_fwd(c);
            })
            .set_fulfill([&] (Cluster* c){
                if (c->merge_lvl() ==  c->lvl()){
                    fwd_geqrt.fulfill_promise(c);
                }
                else if (c->col_interior_deps == 0) {
                    fwd_spars.fulfill_promise(c);
                }
                else {
                    for (auto ein: c->edgesIn){
                        if (ein != nullptr && (ein->n1->lvl() == c->merge_lvl())){
                            auto eit = find_if(ein->n1->edgesOut.begin(), ein->n1->edgesOut.end(), [&c](Edge* e){return e->n2 == c;});
                            assert(eit != ein->n1->edgesOut.end());
                            fwd_tsqrt.fulfill_promise(eit);
                        }
                    }
                }
            })
            .set_name([](Cluster* c) {
                return "fwd_merge_" + to_string(c->get_order());
            })
            .set_priority([&](Cluster*) {
                return 5;
            });

        for(auto c: this->interiors[0]){
            fwd_geqrt.fulfill_promise(c);
        }

        for (auto self: this->interfaces[0]){ 
            if (self->col_interior_deps == 0){
                fwd_spars.fulfill_promise(self);
            }
        }
        tp.join();
    }

    // Bwd
    {
        const int ttor_threads = n_threads;
        #ifdef USE_MKL
            mkl_set_num_threads(1);
            mkl_set_dynamic(0); // no dynamic multi-threading
        #else
            openblas_set_num_threads(1);
        #endif

        // Threadpool
        Threadpool_shared tp(ttor_threads, verb, "solve_");
        /* Elimination */
        Taskflow<Cluster*> bwd_geqrt(&tp, verb);
        /* Scale and Sparsify */
        Taskflow<Cluster*> bwd_spars(&tp, verb);
        /* Merge */
        Taskflow<Cluster*> bwd_merge(&tp, verb);

        bwd_geqrt.set_mapping([&] (Cluster* c){
                return (c->get_order()+1)%ttor_threads; 
            })
            .set_indegree([](Cluster* c){
                if (c->edgesIn.size() == 0) return (unsigned long )1; // only at last level and will be seeded
                return c->edgesIn.size(); 
            })
            .set_task([&] (Cluster* c) {
                QR_bwd(c);
            })
            .set_fulfill([&] (Cluster* c){
                // Fulfill merge 
                bwd_merge.fulfill_promise(c);
            })
            .set_name([](Cluster* c) {
                return "bwd_geqrt_" + to_string(c->get_order());
            })
            .set_priority([&](Cluster*) {
                return 6;
            });

        bwd_merge.set_mapping([&] (Cluster* c){
                return c->get_order()%ttor_threads; 
            })
            .set_indegree([&](Cluster* c){
                return 1;
            })
            .set_task([&] (Cluster* c) {
                merge_bwd(c);
            })
            .set_fulfill([&] (Cluster* c){
                // Fulfill bwd_spars on the children 
                for (auto child: c->children){
                    bwd_spars.fulfill_promise(child);
                }
            })
            .set_name([](Cluster* c) {
                return "bwd_merge_" + to_string(c->get_order());
            })
            .set_priority([&](Cluster*) {
                return 5;
            });

        bwd_spars.set_mapping([&] (Cluster* c){
                return c->get_order()%ttor_threads; 
            })
            .set_indegree([&](Cluster* c){
                return 1;
            })
            .set_task([&] (Cluster* c) {
                // Apply Q from spars and R from scaling
                if (want_sparsify(c)){
                    orthogonalD_bwd(c); // BWD sparsification
                    scaleD_bwd(c); // BWD scaling
                }
            })
            .set_fulfill([&] (Cluster* c){
                // Fulfill bwd_qr on interiors that depend on c
                for (auto eout: c->edgesOut){
                    if (eout != nullptr && (eout->n2->lvl() == c->merge_lvl())){
                        bwd_geqrt.fulfill_promise(eout->n2);
                    }
                }
                // BWD merge at the previous level
                if (c->merge_lvl()>0) bwd_merge.fulfill_promise(c);
            })
            .set_name([](Cluster* c) {
                return "bwd_spars_" + to_string(c->get_order());
            })
            .set_priority([&](Cluster*) {
                return 5;
            });

        for(auto c: this->interiors[nlevels-1]){
            bwd_geqrt.fulfill_promise(c);
        }

        tp.join();
    }

    // Extract solution
    for(auto cluster : bottom_original()) {
        cluster->extract_vector(x);
    }

    // Permute back
    x = this->cperm.asPermutation() * x; 
}