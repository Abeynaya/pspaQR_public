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

/* Add an edge between c1 and c2
 such that A(c2, c1) block is non-zero */
Edge* ParTree::new_edgeOut(Cluster* c1, Cluster* c2){
    Edge* e;

    MatrixXd* A = new MatrixXd(c2->rows(), c1->cols());
    A->setZero();
    e = new Edge(c1, c2, A);
    c1->cnbrs.insert(c2);
    c1->add_edgeOut(e);

    if (c1->get_level()>this->current_bottom && c2->get_level()>this->current_bottom && c1 != c2){
        c1->add_edge_spars_out(e);
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

    if (c1->get_level()>this->current_bottom && c2->get_level()>this->current_bottom && c1 != c2){
        c1->add_edge_spars_out(e);
    }
    return e; 
}

void ParTree::alloc_fillin(Cluster* c, Cluster* n, Taskflow<Edge*>* add_edge_tf){
    list<Edge*> n_edges;
    for (auto edge: c->edgesOut){
        Cluster* nbr_c = edge->n2; 
        auto found_out = find_if(n->edgesOut.begin(), n->edgesOut.end(), [&nbr_c](Edge* e){return (e->n2 == nbr_c );});
        auto found_fillin = n->edgesOutFillin.end();
        if (found_out == n->edgesOut.end()){
            found_fillin = find_if(n->edgesOutFillin.begin(), n->edgesOutFillin.end(), [&nbr_c](Edge* e) {return (e->n2 == nbr_c);});
            if (found_fillin == n->edgesOutFillin.end()){ // Cannot find the edge
                Edge* ecn = new_edgeOutFillin(n, nbr_c);
                ecn->interior_deps++;
                add_edge_tf->fulfill_promise(ecn); // add edgeInFillin
                n_edges.push_back(ecn);
            }
            else {
                (*found_fillin)->interior_deps++;
                n_edges.push_back(*found_fillin);
            }
        }
        else {
            (*found_out)->interior_deps++;
            n_edges.push_back(*found_out);
        }
    }
    n->interiors2edges[c->get_order()] = n_edges;
    return;
}

/* Elimination */
// Only compute Householder reflection
int ParTree::house(Cluster* c){
    if (c->cols()==0){
        c->set_eliminated();
        cout << "cluster has zero cols" << endl;
        exit(1);
    }

    // Combine edgesIn and edgesInFillin -- just to be consistent
    c->combine_edgesIn();

    /* Do Householder reflections */
    vector<MatrixXd*> H(c->edgesOut.size());
    int num_rows=0;
    int num_cols=0;
    int count=0;
    for (auto edge: c->edgesOut){
        assert(edge->A21 != nullptr);
        assert(edge->n2->is_eliminated() == false);
        H[count]= edge->A21;
        num_rows += edge->A21->rows();
        num_cols = edge->A21->cols(); // Must be the same
        count++;
    }
    assert(num_rows>0);
    assert(num_cols>0);

    MatrixXd Q = MatrixXd::Zero(num_rows, num_cols);
    VectorXd tau = VectorXd::Zero(num_cols);
    MatrixXd T = MatrixXd::Zero(num_cols, num_cols);

    concatenate(H, &Q);
    geqrf(&Q, &tau);
    larft(&Q, &tau, &T);

    c->set_Q(Q);
    c->set_tau(tau);
    c->set_T(T);
    c->set_eliminated();
    return 0;
}

// Update Q^T A after householder on interiors
int ParTree::update_cluster(Cluster* c, Cluster* n, Taskflow<Cluster*>* scale_geqrf_tf, Taskflow<Edge*>* scale_larfb_trsm_tf){
    n->combine_edgesOut();
    // Get all the edges that will be modified
    auto n_edges = n->interiors2edges[c->get_order()];

    // Concatenate to get the matrix V so that V <- Q^T V
    vector<MatrixXd*> N(n_edges.size());
    MatrixXd V(c->get_Q()->rows(), n->cols());
    int curr_row = 0;
    int count=0;
    for (auto ecn: n_edges){
        N[count]=ecn->A21;
        V.middleRows(curr_row, ecn->A21->rows()) = *ecn->A21; 
        curr_row += ecn->A21->rows();  
        count++;
    }

    // Compute V = Q^T V
    larfb(c->get_Q(), c->get_T(), &V);   

    // Copy V back to the edges struct
    curr_row = 0;
    for (int i=0; i< N.size(); ++i){
        *(N[i]) = V.middleRows(curr_row, N[i]->rows());
        curr_row += N[i]->rows();
    } 

    // Fullfill dependencies 
    if (this->ilvl >= skip && this->ilvl < nlevels-2  && this->tol != 0){
        for (auto edge: n_edges){
            if (edge->n2 != c){
                if (edge->n2 == n) scale_geqrf_tf->fulfill_promise(n); // Scale GEQRF
                else scale_larfb_trsm_tf->fulfill_promise(edge);
            }
        }
    }
    return 0;
}
/* Scaling */
void ParTree::geqrf_cluster(Cluster* c){
    Edge* e = c->self_edge();
    assert(e != nullptr);
    MatrixXd Q = *(e->A21);
    VectorXd t = VectorXd::Zero(c->cols());
    MatrixXd T = MatrixXd::Zero(c->cols(), c->cols());

    geqrf(&Q, &t);
    larft(&Q, &t, &T);

    // Setting this is important
    (*e->A21).topRows(c->cols()) = MatrixXd::Identity(c->cols(), c->cols());
    (*e->A21).bottomRows(c->rows()-c->cols()) = MatrixXd::Zero(c->rows()-c->cols(),c->cols());

    c->set_Qs(Q);
    c->set_Ts(T);
    c->set_taus(t);
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

    for (this->ilvl=0; this->ilvl < nlevels; ++ilvl){
        auto lvl0 = systime();
    	int ttor_threads = this->n_threads;
        int blas_threads = this->n_threads / ttor_threads;
        #ifdef USE_MKL
        	mkl_set_num_threads(blas_threads);
        #else
        	openblas_set_num_threads(blas_threads);
        #endif
        if (verb) printf("  Threads count: total %d, ttor %d, BLAS %d\n", this->n_threads, ttor_threads, blas_threads);

        timeval vstart, vend, estart, eend;
        // Initializing it incase sparsification is skipped
        timeval spstart=systime();
        timeval spend=spstart;
        {
            // Threadpool
            Threadpool_shared tp(this->n_threads, verb, "get_spars_");
            Taskflow<Cluster*> get_sparsity_tf(&tp, verb);
            Taskflow<pCluster2> fillin_tf(&tp, verb);
            Taskflow<Edge*> addEdgeIn_tf(&tp, verb);


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
                    if (verb) cout << cn[0]->get_id() << " " << cn[1]->get_id() << endl;
                    alloc_fillin(cn[0], cn[1], &addEdgeIn_tf);
                })
                .set_name([](pCluster2 cn) {
                    return "fillin_" + to_string(cn[0]->get_order()) + "_" + to_string(cn[1]->get_order());
                })
                .set_priority([&](pCluster2) {
                    return 7;
                });

            // Fill-in edges 
            addEdgeIn_tf.set_mapping([&] (Edge* e){
                    return e->n2->get_order()%ttor_threads; // Mapping matters to avoid race conditions -- IMPORTANT
                })
                .set_binding([] (Edge*) {
                    return true;
                })
                .set_indegree([](Edge*){
                    return 1;
                })
                .set_task([&] (Edge* e) { // Will work because of binding set to true
                    Cluster* c1 = e->n1;
                    Cluster* c2 = e->n2;
                    // Sufficient to check only in edgeInFillin struct
                    auto found = find_if(c2->edgesInFillin.begin(), c2->edgesInFillin.end(), [&c1](Edge* in_edge){return in_edge->n1 == c1;});
                    if (found == c2->edgesInFillin.end()){
                        c2->add_edgeInFillin(e);
                        if (c1->get_level()>this->ilvl && c2->get_level()>this->ilvl && c1 != c2){
                            c2->add_edge_spars_in(e);
                        }
                    }
                })
                .set_name([](Edge* e) {
                    return "addEdgeIn_" + to_string(e->n1->get_order())+"_"+to_string(e->n2->get_order());
                })
                .set_priority([&](Edge* ) {
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

        }

        {
            // Threadpool
            Threadpool_shared tp(this->n_threads, verb, "elmn_");
            /* Elimination */
            Taskflow<Cluster*> elmn_house_tf(&tp, verb);
            Taskflow<pCluster2> elmn_applyQ_tf(&tp, verb);
            /* Scale */
            Taskflow<Cluster*> scale_geqrf_tf(&tp, verb);
            Taskflow<Edge*> scale_larfb_trsm_tf(&tp, verb);
            /* Sparsify */
            Taskflow<Cluster*> sparsify_rrqr_tf(&tp, verb);
            Taskflow<Edge*> sparsify_larfb_tf(&tp, verb);

            Logger log(10000000); 
            
            if (this->ttor_log){
                // log = Logger(10000000);
                tp.set_logger(&log);
            }


            elmn_house_tf.set_mapping([&] (Cluster* c){
                    return c->get_order()%ttor_threads; 
                })
                .set_indegree([](Cluster*){
                    return 1;
                })
                .set_task([&] (Cluster* c) {
                    if(verb) cout << c->get_id() << endl;
                    house(c);
                })
                .set_fulfill([&] (Cluster* c){
                    for (auto r: c->rsparsity){
                        elmn_applyQ_tf.fulfill_promise({c, r});
                    }
                })
                .set_name([](Cluster* c) {
                    return "elmn_house_" + to_string(c->get_order());
                })
                .set_priority([&](Cluster*) {
                    return 6;
                });

            elmn_applyQ_tf.set_mapping([&] (pCluster2 cn){
                    return cn[1]->get_order()%ttor_threads; // Important to avoid race conditions 
                })
                .set_binding([] (pCluster2) {
                    return true;
                })
                .set_indegree([](pCluster2 cn){
                    return 1;  // Householder on c
                })
                .set_task([&] (pCluster2 cn) {
                    if (verb) cout << cn[0]->get_id() << " " << cn[1]->get_id() << endl;
                    update_cluster(cn[0], cn[1], &scale_geqrf_tf, &scale_larfb_trsm_tf);
                })
                .set_name([](pCluster2 cn) {
                    return "elmn_applyQ_" + to_string(cn[0]->get_order()) + "_" + to_string(cn[1]->get_order());
                })
                .set_priority([&](pCluster2) {
                    return 7;
                });

            /* Scaling */
            scale_geqrf_tf.set_mapping([&] (Cluster* c){
                    return c->get_order()%ttor_threads; 
                })
                .set_indegree([&](Cluster* c){
                    if (c->self_edge()->interior_deps==0) return (unsigned long)1;
                    return c->self_edge()->interior_deps;
                })
                .set_task([&] (Cluster* c) {
                    // Combine edgesIn and edgesInFillin
                    c->combine_edgesIn();
                    if(want_sparsify(c)) geqrf_cluster(c);
                })
                .set_fulfill([&] (Cluster* c){
                    // Fulfill these even for clusters that are not sparsified
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
                    sparsify_rrqr_only(c);
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
                    elmn_house_tf.fulfill_promise(self);
                }

                for (auto self: this->interfaces_current()){ 
                    if (self->self_edge()->interior_deps == 0){
                        scale_geqrf_tf.fulfill_promise(self);
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
        // {
        //     this->current_bottom++;
        //     Threadpool_shared tp(this->n_threads, verb, "merge_");
        //     Taskflow<Cluster*> update_sizes_tf(&tp, verb);
        //     Taskflow<pCluster2> update_edges_tf(&tp, verb);
            

        //     Logger log(10000000); 
        //     if (this->ttor_log){
        //         tp.set_logger(&log);
        //     }


        //     /* Scaling */
        //     update_sizes_tf.set_mapping([&] (Cluster* c){
        //             return c->get_order()%ttor_threads; 
        //         })
        //         .set_indegree([](Cluster*){
        //             return 1;
        //         })
        //         .set_task([&] (Cluster* c) {
        //             parent_edges(c);
        //         })
        //         .set_fulfill([&] (Cluster* c){
                    
        //         })
        //         .set_name([](Cluster* c) {
        //             return "scale_geqrf_" + to_string(c->get_order());
        //         })
        //         .set_priority([&](Cluster*) {
        //             return 5;
        //         });

        // }

        // Keep merge sequential for now
        auto mstart = systime();
        {   
            if (this-> ilvl < nlevels-1){
                merge_all();
            }
        }
        auto mend = systime();
        auto lvl1 = systime();

        cout << "lvl: " << ilvl << "    " ;  
        cout << fixed << setprecision(3)  
             << "find fill-in: " <<   elapsed(vstart,vend) << "  "
             << "elmn: " <<   elapsed(estart,eend) << "  "
        //      << "shift: " <<  elapsed(shstart,shend) << "  "
             // << "scale: " << elapsed(scstart,scend) << "  "
             // << "sprsfy: " << elapsed(spstart,spend) << "  "
             << "merge: " << elapsed(mstart,mend) << "  " 
             << "Time per level: " << elapsed(lvl0,lvl1) << " "
        //      << "size top_sep: " <<  get<0>(topsize()) << ", " << get<1>(topsize()) << "   "
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
    MatrixXd* Q = c->get_Q();
    VectorXd xc = VectorXd::Zero(Q->rows());
    int curr_row = 0;
    int i=0;
    for (auto e: c->edgesOut){
        if (e->A21 != nullptr){ // can be null for rect matrices
            int s = e->A21->rows();
            xc.segment(curr_row, s) = e->n2->get_x()->segment(0, s);
            curr_row += s;
            ++i;
        }
    }
    larfb(Q, c->get_T(), &xc);
    curr_row = 0;
    i=0;
    for (auto e: c->edgesOut){
        if (e->A21 != nullptr){
            int s = e->A21->rows(); // will not be affected even if e->n2->rows() change b/c of sparsification
            e->n2->get_x()->segment(0, s) = xc.segment(curr_row, s);
            curr_row += s;
            ++i;
        }
    }
}

void ParTree::QR_bwd(Cluster* c) const{
    int i=0;
    Segment xs = c->get_x()->segment(0, c->cols());
    for (auto e: c->edgesIn){
        assert(e->A21 != nullptr);
        int s = e->A21->cols();
        xs -= (e->A21->topRows(c->cols()))*(e->n1->get_x()->segment(0, s)); 
        ++i;    
    }
    // MatrixXd R = c->self_edge()->A21->topRows(xs.size()); // A21 block not updated anymore
    MatrixXd R = c->get_Q()->topRows(xs.size());
    trsv(&R, &xs, CblasUpper, CblasNoTrans, CblasNonUnit);
}

void ParTree::scaleD_fwd(Cluster* c) const{
    Segment xs = c->get_x()->segment(0,c->original_rows());
    ormqr_trans(c->get_Qs(), c->get_taus(), &xs);
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
// // Sequential for now
void ParTree::solve(VectorXd b, VectorXd& x) const{
    // Permute the rhs according to this->rperm
    b = this->rperm.asPermutation().transpose()*b;

    // Set solution
    for (auto cluster: bottom_original()){
        cluster->set_vector(b);
    }

    // Fwd
    for (int l=0; l<nlevels; ++l){
        // Elmn
        for (auto self: this->interiors[l]){
            QR_fwd(self);
        }
        // Scaling 
        if (l>=skip && l< nlevels-2  && this->tol != 0){
            for (auto self: this->interfaces2sparsify[l]){
                scaleD_fwd(self);
            }
        }
        // Sparsification
        if (l>=skip && l< nlevels-2  && this->tol != 0){
            for (auto self: this->interfaces2sparsify[l]){
                orthogonalD_fwd(self);
            }
        }
        
        // Merge
        if (l < nlevels-1){
            for (auto self: this->bottoms[l+1]){
                merge_fwd(self);
            }
        }

    }

    // Bwd
    for (int l=nlevels-1; l>=0; --l){
        // Sparsification
        if (l>=skip && l< nlevels-2  && this->tol != 0){
            for (auto self: this->interfaces2sparsify[l]){
                orthogonalD_bwd(self);
            }
        }

        // Scaling 
        if (l>=skip && l< nlevels-2  && this->tol != 0){
            for (auto self: this->interfaces2sparsify[l]){
                scaleD_bwd(self);
            }
        }
        // Elmn
        for (auto self: this->interiors[l]){
            QR_bwd(self);
        }
        
        // Merge
        if (l > 0){
            for (auto self: this->bottoms[l]){
                merge_bwd(self);
            }
        }
    }
    // Extract solution
    for(auto cluster : bottom_original()) {
        cluster->extract_vector(x);
    }

    // Permute back
    x = this->cperm.asPermutation() * x; 
}