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
int ParTree::nranks() const{return n_ranks;}

/* Helper function */
// EdgeIt find_out_edge(Cluster* m, Cluster* n){
//     auto found = find_if(m->edgesOut.begin(), m->edgesOut.end(), [&n](Edge* e_m){return e_m->n2 == n;});
//     assert(found != m->edgesOut.end());
//     // if (found == m->edgesOut.end()){
//     //     found = find_if(m->edgesOutFillin.begin(), m->edgesOutFillin.end(), [&n](Edge* e_m){return e_m->n2 == n;});
//     //     assert( found!= m->edgesOutFillin.end()); // Has to exist
//     // }
//     return found;
// }

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

Cluster* ParTree::get_interior(int c_order) const {
    auto found = find_if(this->interiors_current().begin(), this->interiors_current().end(), [&c_order](Cluster* n){return c_order == n->get_order();});
    if (found == this->interiors_current().end()){
        cout << my_rank << " " << current_bottom << " " << c_order << endl;
    }
    assert(found != this->interiors_current().end()); 
    return *found; 
}

Cluster* ParTree::get_interface(int c_order) const {
    auto found = find_if(this->interfaces_current().begin(), this->interfaces_current().end(), [&c_order](Cluster* n){return c_order == n->get_order();});
    if (found == this->interfaces_current().end()){
        cout << my_rank << " " << current_bottom << " " << c_order << endl;
    }
    assert(found != this->interfaces_current().end()); 
    return *found; 
}

Cluster* ParTree::get_cluster(int c_order) const {
    auto found = find_if(this->bottom_current().begin(), this->bottom_current().end(), [&c_order](Cluster* n){return c_order == n->get_order();});
    if (found == this->bottom_current().end()){
        cout << my_rank << " " << current_bottom << " " << c_order << endl;
    }
    assert(found != this->bottom_current().end()); 
    return *found; 
}


int ParTree::get_rank(int part, int mergelvl){
    int nparts_lvl = nparts(); 
    for (int lvl=0; lvl < mergelvl; lvl++){
        nparts_lvl /= 2;
    }

    int rank_lvl = 0;
    if(nparts_lvl > nranks()) {
        int parts_per_rank = nparts_lvl / nranks();
        rank_lvl = (part / parts_per_rank);
    } else { 
        rank_lvl = part;
    }
    assert(rank_lvl >= 0);
    assert(rank_lvl < nranks());
    return rank_lvl;
}

int ParTree::cluster2rank(Cluster* c){
    int mergelvl = c->merge_lvl();
    int part = c->section();
    return get_rank(part, mergelvl);
}

int ParTree::edge2rank(Edge* e) {
    return cluster2rank(e->n1);
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
    if(c1 != c2) {
        int dest = cluster2rank(c2);
        if (dest == my_rank) c2->add_edgeIn(e,is_spars_nbr); // thread-safe using lock_guard
        // else {
        //     // A.M of an empty edge to dest
        //     ...
        // }
    }
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
    if (c1 != c2) { // If distributed... c2 can reside in a different rank
        c2->add_edgeInFillin(e,is_spars_nbr); // thread-safe using lock_guard
        // else {
        //     // AM send edge->n1 and edge->n2 
        // }
    }
    return e; 
}

void ParTree::alloc_fillin(Cluster* c, Cluster* n, map<int,vector<int>>& edges_to_send){
    for (auto edge: c->edgesOut){
        Cluster* nbr_c = edge->n2; 
        auto found_out = find_if(n->edgesOut.begin(), n->edgesOut.end(), [&nbr_c](Edge* e){return (e->n2 == nbr_c );});
        auto found_fillin = n->edgesOutFillin.end();
        if (found_out == n->edgesOut.end()){
            found_fillin = find_if(n->edgesOutFillin.begin(), n->edgesOutFillin.end(), [&nbr_c](Edge* e) {return (e->n2 == nbr_c);});
            if (found_fillin == n->edgesOutFillin.end()){ // Cannot find the edge
                // Edge* ecn = new_edgeOutFillin(n, nbr_c);
                Edge* ecn = new_edgeOut(n, nbr_c);
                if (cluster2rank(nbr_c)!=my_rank) edges_to_send[cluster2rank(nbr_c)].push_back(nbr_c->get_order());
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
    // c->combine_edgesIn(); // No race condition

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
    Cluster* c = e->n2;
    assert(c->is_eliminated());

    // int nrows = e->A21->rows();
    // int ncols = e->A21->cols();

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
void ParTree::compute_new_edges(Cluster* snew, map<int,vector<int>>& edges_to_send){
    set<Cluster*> edges_merged;
    for (auto sold: snew->children){
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
        else if (snew->lvl() == snew->merge_lvl()) n->col_interior_deps++; // Needed for solve
        // If new edgeIn has to be sent to a different rank
        if (cluster2rank(n)!= my_rank) edges_to_send[cluster2rank(n)].push_back(n->get_order());
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
                assert(eold->A21->rows() == nold->rows());
                assert(eold->A21->cols() == sold->cols());

                assert(nold->rposparent >= 0);
                assert(sold->cposparent >= 0);

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

        // Find fill-in and allocate memory -- TODO: Make this sequential (seems easier and more efficient as this is very cheap)
        // All ranks do this for now --- CHANGE: TODO
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
            Communicator comm(MPI_COMM_WORLD);
            Threadpool_dist tp(ttor_threads, &comm, verb, "fillin_"+ to_string(ilvl)+"_" +to_string(my_rank)+"_");
            Taskflow<Cluster*> get_sparsity_tf(&tp, verb);
            Taskflow<pCluster2> fillin_tf(&tp, verb);

            Logger log(1000); 
            if (this->ttor_log){
                tp.set_logger(&log);
            }

            auto send_edge_am = comm.make_active_msg([&] (int& n1_order,  view<int>& n2_orders){
                Cluster* n1 = get_cluster(n1_order);
                for (auto ord : n2_orders){
                    Cluster* n2 = get_cluster(ord);
                    assert(cluster2rank(n2) == my_rank);
                    bool is_spars_nbr = (n1->lvl()>this->current_bottom) && (n2->lvl()>this->current_bottom) && (n1 != n2);
                    n2->add_edgeIn(new Edge(n1, n2), is_spars_nbr); // e->A21 = new MatrixXd(0,0) -- thread safe
                }
            });

            auto send_fulfill_am = comm.make_active_msg([&] (int& c_order, view<int>& c_edges, view<int>& fulfill_deps){
                Cluster* c = get_cluster(c_order);
                // Create c->edgesOut 
                assert(cluster2rank(c)!= my_rank);
                for (auto n_order: c_edges){
                    Cluster* n = get_cluster(n_order);
                    Edge* e = new Edge(c,n);
                    c->edgesOut.push_back(e); // No race condition possible here
                    c->create_T(n_order);
                    if (n_order == c_order) c->add_self_edge(e);
                }
                c->sort_edgesOut(); // important

                // Fulfill deps 
                for (auto n_order: fulfill_deps){
                    Cluster* n = get_cluster(n_order);
                    assert(cluster2rank(n)== my_rank);
                    fillin_tf.fulfill_promise({c,n});
                }
               
            });

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
                    int c_order = c->get_order();
                    map<int,vector<int>> to_send;
                    for (auto n: c->rsparsity){
                        int dest = cluster2rank(n);
                        if (dest == my_rank) fillin_tf.fulfill_promise({c,n});
                        else to_send[dest].push_back(n->get_order());
                    }

                    if (to_send.size()>0){
                        vector<int> out_edges;
                        for (auto e:c->edgesOut){
                            out_edges.push_back(e->n2->get_order());
                        }
                        auto c_edges = view<int>(out_edges.data(), out_edges.size());
                        for(auto& r: to_send){
                            auto fulfill_deps = view<int>(r.second.data(), r.second.size());
                            send_fulfill_am->send(r.first, c_order, c_edges, fulfill_deps);
                        }
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
                    map<int,vector<int>> to_send;
                    alloc_fillin(cn[0], cn[1], to_send);
                    int n1_order = cn[1]->get_order();
                    for(auto& r: to_send){
                        auto edges_view = view<int>(r.second.data(), r.second.size());
                        send_edge_am->send(r.first, n1_order, edges_view);
                    }
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
                    if (cluster2rank(self)==my_rank) get_sparsity_tf.fulfill_promise(self);
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
            Communicator comm(MPI_COMM_WORLD);
            Threadpool_dist tp(ttor_threads, &comm, verb, "elmn_" + to_string(ilvl)+"_"+to_string(my_rank)+"_");
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
            auto geqrt_2_larfb_am = comm.make_active_msg(
                [&] (view<double>& U_data, view<double>& T_data, view<int>& larfb_deps, int& c_order, int& crows, int& ccols){
                    Cluster* c = get_interior(c_order);
                    c->set_eliminated();
                    Edge* c_self = c->self_edge();
                    assert(c_self->A21 != nullptr);
                    // if (c_self->A21 == nullptr) c_self->A21 = new MatrixXd(crows, ccols);
                    // c->create_T(c->get_order());
                    MatrixXd Tmat = Map<MatrixXd>(T_data.data(), ccols, ccols);
                    *(c_self->A21) = Map<MatrixXd>(U_data.data(), crows, ccols);
                    c->set_T(c->get_order(), Tmat);
                    for (int norder: larfb_deps){
                        Cluster* n = get_interface(norder);
                        auto eit = n->find_out_edge(c_order);
                        assert(cluster2rank((*eit)->n1) == my_rank);
                        larfb_tf.fulfill_promise(eit);
                    }
                });

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
                    map<int,vector<int>> send_to;
                    for (EdgeIt it=c->edgesIn.begin(); it!=c->edgesIn.end(); ++it){ // All the edges that need to be updated
                        int dest = edge2rank(*it);
                        if(dest == my_rank) larfb_tf.fulfill_promise(it);
                        else {
                            send_to[dest].push_back((*it)->n1->get_order());
                        }
                    }

                    if (send_to.size() > 0){
                        int crows = c->rows();
                        int ccols = c->cols();
                        int corder = c->get_order();

                        auto U_view = view<double>(c->self_edge()->A21->data(), crows*ccols);
                        auto T_view = view<double>(c->get_T(c->get_order())->data(), ccols*ccols);
                        
                        for (auto& r: send_to ){
                            auto larfb_deps = view<int>(r.second.data(), r.second.size());
                            geqrt_2_larfb_am->send(r.first, U_view, T_view, larfb_deps, corder, crows, ccols);
                        }
                    }

                    // First tsqrt -- same rank
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
                    // First ssrfb -- happens on the same rank
                    auto c_edge_it = c->edgesOut.begin();
                    assert((*c_edge_it)->n2 == c); // first edgeout is c itself when c is at leaf level
                    c_edge_it++;

                    if (c_edge_it != c->edgesOut.end()){
                        Edge* eout = *c_edge_it;
                        Cluster*  n = eout->n2;
                        auto found = m->find_out_edge(n->get_order());
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

                    // First fulfill scale -- same rank
                    Edge* edge_mn = *it[2];
                    if (this->ilvl >= skip && this->ilvl < nlevels-2  && this->tol != 0){
                        if (edge_mn->n2 == edge_mn->n1) scale_geqrt_tf.fulfill_promise(edge_mn->n1); // Scale GEQRF
                        else scale_larfb_trsm_tf.fulfill_promise(edge_mn);
                    }

                    // Next ssrfb -- also on same rank
                    c_edge_it++;
                    Cluster* m = (*it[1])->n1;
                    if (c_edge_it != c->edgesOut.end()){
                        Edge* eout = *c_edge_it;
                        Cluster*  n = eout->n2;
                        auto found = m->find_out_edge(n->get_order());
                        int dest = edge2rank(*found);
                        assert(dest == my_rank); 
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

            auto tsqrt_2_ssrfb_am = comm.make_active_msg(
                [&] (view<double>& U_data, view<double>& T_data, view<int>& ssrfb_deps, int& c_order, int& n_order, int& n_rows, int& c_cols){
                    Cluster* c = get_interior(c_order);
                    // Cluster* n = get_interface(n_order); 

                    // Store the U and T matrices
                    auto eit_cn = find_if(c->edgesOut.begin(), c->edgesOut.end(), [&n_order](Edge* e){return e->n2->get_order() == n_order;});
                    assert(eit_cn != c->edgesOut.end());
                    Edge* e_cn = *eit_cn;
                    assert(e_cn->A21 != nullptr);
                    // if (e_cn->A21 == nullptr) e_cn->A21 = new MatrixXd(n_rows, c_cols);
                    *(e_cn->A21) = Map<MatrixXd>(U_data.data(), n_rows, c_cols);

                    int nb = min(32, c_cols);
                     // c->create_T(n_order);
                    MatrixXd Tmat = Map<MatrixXd>(T_data.data(), nb, c_cols);
                    c->set_T(n_order, Tmat);

                    // Satisfy dependencies 
                    for (int m_order: ssrfb_deps){
                        Cluster* m = get_interface(m_order);
                        auto eit_mn = m->find_out_edge(n_order);
                        auto eit_mc = m->find_out_edge(c_order);
                        assert(cluster2rank(m)==my_rank);
                        ssrfb_tf.fulfill_promise({eit_cn, eit_mc, eit_mn}); // c->n, m->c, m->n
                    }
                });

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
                    // Next tsqrt -- same rank
                    if (eit_next != c->edgesOut.end()){
                        tsqrt_tf.fulfill_promise(eit_next);
                    }
                    // All ssrfb in its row
                    map<int,vector<int>> send_to;
                    for (EdgeIt it=c->edgesIn.begin(); it!=c->edgesIn.end(); ++it){ // All the edges that need to be updated
                        Cluster* m = (*it)->n1;
                        int dest = cluster2rank(m);
                        if (dest == my_rank){
                            auto found = m->find_out_edge(n->get_order());
                            ssrfb_tf.fulfill_promise({eit, it, found}); // c->n, m->c, m->n
                        }
                        else {
                            send_to[dest].push_back(m->get_order());
                        }
                    }

                    if (send_to.size()>0){
                        auto c_order = c->get_order();
                        auto n_order = n->get_order();
                        int nrows = n->rows();
                        auto ccols = c->cols();
                        int nb = min(32, ccols);
                        auto U_view = view<double>((*eit)->A21->data(), nrows*ccols);
                        auto T_view = view<double>(c->get_T(n->get_order())->data(), nb*ccols);
                        for (auto& r: send_to){
                            auto ssrfb_deps = view<int>(r.second.data(), r.second.size());
                            tsqrt_2_ssrfb_am->send(r.first, U_view, T_view, ssrfb_deps, c_order, n_order, nrows, ccols);
                        }
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
                    if (cluster2rank(self) == my_rank) {
                        geqrt_tf.fulfill_promise(self);
                    }
                }

                for (auto self: this->interfaces_current()){ 
                    if (this->ilvl < nlevels-2  && this->ilvl >= skip && this->tol != 0){
                        if (self->self_edge()->interior_deps == 0){
                            if (cluster2rank(self)==my_rank) scale_geqrt_tf.fulfill_promise(self);
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
            {
                // Send all interfaces 

                Communicator comm(MPI_COMM_WORLD);
                Threadpool_dist tp(ttor_threads, &comm, verb, "merge_send_" + to_string(ilvl)+"_"+to_string(my_rank)+"_");

                auto send_size_am = comm.make_active_msg([&] (view<int3>& sizes_view){
                    for (auto orc : sizes_view){
                        int order = orc[0];
                        Cluster* c = get_interface(order);
                        c->reset_size(orc[1], orc[2]);
                    }
                });

                auto send_edge_am = comm.make_active_msg([&] (int& n1_order, int& n2_order, int& A_rows, int& A_cols, view<double>& A){
                    Cluster* n1 = get_interface(n1_order);
                    Cluster* n2 = get_interface(n2_order);
                    MatrixXd* A21 = new MatrixXd(A_rows, A_cols);
                    *A21 = Map<MatrixXd>(A.data(), A_rows, A_cols);
                    auto found = find_if(n1->edgesOut.begin(), n1->edgesOut.end(), [& n2_order](Edge* e){return e->n2->get_order() == n2_order;});
                    if (found == n1->edgesOut.end()){
                        n1->add_edgeOut_threadsafe(new Edge(n1, n2, A21)); // thread-safe
                    };
                });

                vector<int3> to_send;
                for (auto sold: this->interfaces_current()){
                    if (cluster2rank(sold) == my_rank){ // Contains the most up-to-date info
                        // Send to all ranks
                        to_send.push_back({sold->get_order(),sold->rows(),sold->cols()});
                    }
                }

                auto sizes_view = view<int3>(to_send.data(), to_send.size());
                for (int dest=0; dest < nranks(); ++dest){
                    if(dest!=my_rank) {
                        send_size_am->send(dest, sizes_view);
                    }
                }

                for (auto sold: this->interfaces_current()){
                    if (cluster2rank(sold) == my_rank){
                        int dest = cluster2rank(sold->get_parent());
                        if (dest != my_rank){
                            for (auto e: sold->edgesOut){
                                if (!(e->n2->is_eliminated())){
                                    int A_rows = e->A21->rows();
                                    int A_cols = e->A21->cols();
                                    auto A_view = view<double>(e->A21->data(), A_rows*A_cols);
                                    int n1_order = e->n1->get_order();
                                    int n2_order = e->n2->get_order();
                                    send_edge_am->send(dest, n1_order, n2_order, A_rows, A_cols, A_view);
                                }
                            }
                        }
                    }
                }
                tp.join();
            }

            // Actual merge
            this->current_bottom++;
            // Update sizes -- sequential because we need a synch point after this and this is cheap anyway
            for (auto snew: this->bottom_current()){
                // if (cluster2rank(snew) == my_rank){ // Need to be done in all ranks
                    int rsize = 0;
                    int csize = 0;
                    for (auto sold: snew->children){
                        sold->rposparent = rsize;
                        sold->cposparent = csize;
                        rsize += sold->rows();
                        csize += sold->cols();
                        for (auto d2c: sold->dist2connxs) if(d2c->lvl() >= this->current_bottom) snew->dist2connxs.insert(d2c->get_parent());
                    }
                    snew->set_size(rsize, csize); 
                    snew->set_org(rsize, csize);
            }

            // Update edges
            Communicator comm(MPI_COMM_WORLD);
            Threadpool_dist tp(ttor_threads, &comm, verb, "merge_" + to_string(ilvl)+"_"+to_string(my_rank)+"_");
            Taskflow<Cluster*> update_edges_tf(&tp, verb);
            
            Logger log(1000); 
            if (this->ttor_log){
                tp.set_logger(&log);
            }

            auto send_edge_am = comm.make_active_msg([&] (int& n1_order,  view<int>& n2_orders){
                Cluster* n1 = get_cluster(n1_order);
                for (auto ord : n2_orders){
                    Cluster* n2 = get_cluster(ord);
                    assert(cluster2rank(n2) == my_rank);
                    bool is_spars_nbr = (n1->lvl()>this->current_bottom) && (n2->lvl()>this->current_bottom) && (n1 != n2);
                    n2->add_edgeIn(new Edge(n1, n2), is_spars_nbr); // e->A21 = new MatrixXd(0,0) -- thread safe
                }
            });

            /* Compute new edges */
            update_edges_tf.set_mapping([&] (Cluster* c){
                    return (c->get_order()+1)%n_threads; 
                })
                .set_indegree([](Cluster*){
                    return 1;
                })
                .set_task([&] (Cluster* c) {
                    map<int,vector<int>> to_send;
                    compute_new_edges(c, to_send);
                    int c_order = c->get_order();
                    for (auto& r: to_send){
                        auto edges_view = view<int>(r.second.data(), r.second.size());
                        send_edge_am->send(r.first, c_order, edges_view);
                    }
                })
                .set_name([](Cluster* c) {
                    return "update_edges_" + to_string(c->get_order());
                })
                .set_priority([&](Cluster*) {
                    return 5;
                });

            for (auto snew: this->bottom_current()){
                if (cluster2rank(snew) == my_rank) update_edges_tf.fulfill_promise(snew);
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