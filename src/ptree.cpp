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

    if (c1->get_level()>this->ilvl && c2->get_level()>this->ilvl && c1 != c2){
        c1->add_edge_spars_out(e);
    }
    return e; 
}

void ParTree::setup_fillin(Cluster* c, Cluster* n, Taskflow<pEdge2>* add_edge_tf){
    auto found_self = find_if(c->edgesOut.begin(), c->edgesOut.end(), [&c](Edge* e){return e->n2 == c;});
    assert(found_self != c->edgesOut.end()); // Should be there
    
    for (auto r: c->rsparsity){
        auto found = find_if(n->edgesIn.begin(), n->edgesIn.end(), [&r](Edge* e){return (e->n1 == r);});
        if (found == n->edgesIn.end()){
            Edge* e;
            MatrixXd* A = new MatrixXd(n->rows(), r->cols());
            A->setZero();
            e = new Edge(r, n, A);
            if(r != n) {n->add_edgeIn(e);}
            add_edge_tf->fulfill_promise({*found_self,e});
        }
        else {
            add_edge_tf->fulfill_promise({*found_self,*found});
        }
    }
    return;
    
}

void ParTree::alloc_fillin(Cluster* n, Taskflow<pEdge2>* add_edge_tf){
    auto found_int = find_if(this->interiors_current().begin(), this->interiors_current().end(), [&n](Cluster* c){return n==c;});
    if (found_int == this->interiors_current().end()){
        for (auto elm_c: this->interiors_current()){ // Clusters that will be eliminated in the current lvl
            for (auto eout: elm_c->edgesOut){
                if (eout->n2 == n){
                    setup_fillin(elm_c, n, add_edge_tf);
                    break;
                }
            }
        }
    }
    else {// Cluster n is an interior
        setup_fillin(n,n, add_edge_tf);
    }
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

    /* Do Householder reflections */
    vector<MatrixXd*> H;
    int num_rows=0;
    int num_cols=0;
    for (auto edge: c->edgesOut){
        assert(edge->A21 != nullptr);
        assert(edge->n2->is_eliminated() == false);
        H.push_back(edge->A21);
        num_rows += edge->A21->rows();
        num_cols = edge->A21->cols(); // Must be the same
    }
    assert(num_rows>0);
    assert(num_cols>0);

    MatrixXd Q = MatrixXd::Zero(num_rows, num_cols);
    VectorXd tau = VectorXd::Zero(num_cols);
    MatrixXd T = MatrixXd::Zero(num_cols, num_cols);

    concatenate(H, &Q);
    geqrf(&Q, &tau);
    
    // Not really important anymore --- can be removed 
    if (this->square){ // Copy back to H 
        int curr_row = 0;
        for (int i=0; i< H.size(); ++i){
            *(H[i]) = Q.middleRows(curr_row, H[i]->rows());
            curr_row += H[i]->rows();
        }
    }
    else // Don't store Q. Need only R
        *(H[0]) = Q.topRows(H[0]->cols());

    larft(&Q, &tau, &T);
    c->set_Q(Q);
    c->set_tau(tau);
    c->set_T(T);
    c->set_eliminated();
    return 0;
}

// Update Q^T A after householder on interiors
int ParTree::update_cluster(Cluster* c, Cluster* n, Taskflow<Edge*>* tf){
    // // Concatenate the blocks to get Q
    // vector<MatrixXd*> H;
    // int num_rows=0;
    // int num_cols=0;
    // for (auto edge: c->edgesOut){
    //     H.push_back(edge->A21);
    //     num_rows += edge->A21->rows();
    //     num_cols = edge->A21->cols(); // Must be the same
    // }
    // MatrixXd* Q = new MatrixXd(num_rows, num_cols);
    // concatenate(H, Q);


    // Concatenate to get the matrix V so that V <- Q^T V
    vector<MatrixXd*> N;
    MatrixXd V(c->get_Q()->rows(), n->cols());
    int curr_row = 0;

    for (auto edge: c->edgesOut){
        Edge* ecn;
        assert (edge->A21 != nullptr); // Only the cnbrs + c itself
        
        Cluster* nbr_c = edge->n2; 
        auto found_out = find_if(n->edgesOut.begin(), n->edgesOut.end(), [&nbr_c](Edge* e){return (e->n2 == nbr_c );});

        // Fill-in => Allocate memory first
        if (found_out == n->edgesOut.end()){
            ecn = new_edgeOut(n, nbr_c);
            if (n != nbr_c) {
                // insert in edgeIn
                tf->fulfill_promise(ecn);
            }
        }
        else {
            ecn = *(found_out);
        }
        
        assert(ecn->A21 != nullptr);
        N.push_back(ecn->A21);
        V.middleRows(curr_row, ecn->A21->rows()) = *ecn->A21; 
        curr_row += ecn->A21->rows();  
        
    }

    // Compute V = Q^T V
    larfb(c->get_Q(), c->get_T(), &V);   

    // Copy V back to the edges struct
    curr_row = 0;
    for (int i=0; i< N.size(); ++i){
        *(N[i]) = V.middleRows(curr_row, N[i]->rows());
        curr_row += N[i]->rows();
    } 
    return 0;
}
/* Scaling */
void ParTree::geqrf_cluster(Cluster* c){
    if (!want_sparsify(c)) return;

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
    vector<MatrixXd*> Apm; // row nbrs
    vector<MatrixXd*> Amp; // col nbrs
    Edge* e_self = c->self_edge();
    assert(e_self != nullptr);

    int spm=0;
    int smp=0;
    for (auto e: c->edgesInNbrSparsification){
        Apm.push_back(e->A21);
        spm += e->A21->cols();
    }

    for (auto e: c->edgesOutNbrSparsification){
        Amp.push_back(e->A21);
        smp += e->A21->rows();
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

            get_sparsity_tf.set_mapping([&] (Cluster* c){
                    return c->get_order()%ttor_threads; 
                })
                .set_indegree([](Cluster*){
                    return 1;
                })
                .set_task([&] (Cluster* c) {
                    this->get_sparsity(c);
                })
                .set_name([](Cluster* c) {
                    return "get_sparsity_" + to_string(c->get_order());
                })
                .set_priority([&](Cluster*) {
                    return 6;
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
            Taskflow<Cluster*> elmn_house_tf(&tp, verb);
            Taskflow<Edge*> addEdgeIn_tf(&tp, verb);
            Taskflow<pCluster2> elmn_applyQ_tf(&tp, verb);

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
                        if (!(r->is_eliminated())) elmn_applyQ_tf.fulfill_promise({c, r});
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
                .set_indegree([](pCluster2 cn){
                    return 1;  // Householder on c
                })
                .set_task([&] (pCluster2 cn) {
                    if (verb) cout << cn[0]->get_id() << " " << cn[1]->get_id() << endl;
                    update_cluster(cn[0], cn[1], &addEdgeIn_tf);
                })
                .set_binding([] (pCluster2) {
                    return true;
                })
                .set_name([](pCluster2 cn) {
                    return "elmn_applyQ_" + to_string(cn[0]->get_order()) + "_" + to_string(cn[1]->get_order());
                })
                .set_priority([&](pCluster2) {
                    return 7;
                });

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
                    auto found = find_if(c2->edgesIn.begin(), c2->edgesIn.end(), [&c1](Edge* in_edge){return in_edge->n1 == c1;});
                    if (found == c2->edgesIn.end()){
                        c2->add_edgeIn(e);
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
            // Eliminate interiors
            estart = systime();
            {
                for (auto self: this->interiors_current()){ 
                    elmn_house_tf.fulfill_promise(self);
                }
            }
            tp.join();
            eend = systime();

            auto log0 = systime();
            if (this->ttor_log){
                std::ofstream logfile;
                std::string filename = "./logs/elmn_" + to_string(this->ilvl) + ".log." + to_string(0);
                if(this->verb) printf("  Level %d logger saved to %s\n", this->ilvl, filename.c_str());
                logfile.open(filename);
                logfile << log;
                logfile.close();
            }
            auto log1 = systime();
            // cout << "Time to setup_: " << elapsed(tp0,tp1) << endl;
            // cout << "Time to elmn_: " << elapsed(estart,eend) << endl;
            // cout << "Time to write to file elmn_: " << elapsed(log0,log1) << endl;
        }

        // Scaling and Sparsification
        if (this->ilvl >= skip && this->ilvl < nlevels-2  && this->tol != 0){
            assert(this->scale==1);
            Threadpool_shared tp(this->n_threads, verb, "sparsify_");
            /* Scale */
            Taskflow<Cluster*> scale_geqrf_tf(&tp, verb);
            Taskflow<pCluster2> scale_larfb_trsm_tf(&tp, verb);
            /* Sparsify */
            Taskflow<Cluster*> sparsify_rrqr_tf(&tp, verb);
            Taskflow<pCluster2> sparsify_larfb_tf(&tp, verb);

            Logger log(10000000); 
            if (this->ttor_log){
                // log = Logger(10000000);
                tp.set_logger(&log);
            }


            /* Scaling */
            scale_geqrf_tf.set_mapping([&] (Cluster* c){
                    return c->get_order()%ttor_threads; 
                })
                .set_indegree([](Cluster*){
                    return 1;
                })
                .set_task([&] (Cluster* c) {
                    geqrf_cluster(c);
                })
                .set_fulfill([&] (Cluster* c){
                    // Fulfill these even for clusters that are not sparsified
                    // Apply ormqr to rows and cols
                    for (auto ein: c->edgesInNbrSparsification){
                        scale_larfb_trsm_tf.fulfill_promise({c, ein->n1});
                    }
                    for (auto eout: c->edgesOutNbrSparsification){
                        scale_larfb_trsm_tf.fulfill_promise({eout->n2, c});
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
            scale_larfb_trsm_tf.set_mapping([&] (pCluster2 cn){
                    return (cn[0]->get_order() + cn[1]->get_order())%ttor_threads;  
                })
                .set_indegree([](pCluster2){
                    return 2; // the geqrf on clusters c and n
                })
                .set_task([&] (pCluster2 cn) {
                    Cluster* c = cn[0];
                    Cluster* n = cn[1]; 
                    // Find the edge containing A_cn block -- should exist
                    auto found = find_if(c->edgesIn.begin(), c->edgesIn.end(), [&n](Edge* e){return e->n1==n;});
                    assert(found != c->edgesIn.end());
                    Edge* e = *found;
                    if (want_sparsify(c)) larfb(c->get_Qs(), c->get_Ts(), e->A21);
                    if (want_sparsify(n)) trsm_edge(e);

                })
                .set_fulfill([&] (pCluster2 cn){
                    // Sparsify cluster
                    sparsify_rrqr_tf.fulfill_promise(cn[0]);
                    sparsify_rrqr_tf.fulfill_promise(cn[1]);
                })
                .set_name([](pCluster2 cn) {
                    return "scale_apply_" + to_string(cn[0]->get_order())+ "_" + to_string(cn[1]->get_order());
                })
                .set_priority([&](pCluster2) {
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
                        sparsify_larfb_tf.fulfill_promise({c, ein->n1}); // Q_c^T ein->A21
                    }
                    for (auto eout: c->edgesOutNbrSparsification){
                        sparsify_larfb_tf.fulfill_promise({eout->n2, c}); // eout->A21* Q_c
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
            sparsify_larfb_tf.set_mapping([&] (pCluster2 cn){
                    return (cn[0]->get_order()+cn[1]->get_order())%ttor_threads;  
                })
                .set_indegree([](pCluster2){
                    return 2; // the sparsify on clusters c and n
                })
                .set_task([&] (pCluster2 cn) {
                    Cluster* c = cn[0]; 
                    Cluster* n = cn[1]; 
                    // Find the edge containing A_cn block -- should exist
                    auto found = find_if(c->edgesIn.begin(), c->edgesIn.end(), [&n](Edge* e){return e->n1==n;});
                    assert(found != c->edgesIn.end());
                    Edge* e = *found;

                    MatrixXd Aold = *e->A21;
                    // 1. A_cn <- Q_c^T A_cn
                    if (want_sparsify(c)) larfb(c->get_Q_sp(), c->get_T_sp(), &Aold);
                    // 2. A_cn <- A_cn Q_n
                    if (want_sparsify(n)) larfb_notrans_right(n->get_Q_sp(), n->get_T_sp(), &Aold);

                    *e->A21 = Aold.block(0,0,c->get_rank(),n->get_rank());
                })
                .set_name([](pCluster2 cn) {
                    return "sparsify_apply_" + to_string(cn[0]->get_order())+ "_" + to_string(cn[1]->get_order());
                })
                .set_priority([&](pCluster2) {
                    return 3;
                });

            // Scale and sparsify interfaces
            spstart = systime();
            {
                for (auto self: this->interfaces_current()){ 
                    scale_geqrf_tf.fulfill_promise(self);
                }
            }
            tp.join();
            spend = systime();

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

            // cout << "Time to write to file sparsify_: " << elapsed(log0,log1) << endl;

        }

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
             << "sprsfy: " << elapsed(spstart,spend) << "  "
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
    MatrixXd R = c->self_edge()->A21->topRows(xs.size());
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