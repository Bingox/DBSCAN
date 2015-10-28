#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/algorithm/minmax.hpp>
#include <vector>
#include <omp.h>

#include "dbscan.h"

namespace clustering {

	DBSCAN::ClusterData DBSCAN::read_file( std::string & file_name, char delimiter ) {
		size_t features_num = 0, elements_num = 0;
		std::ifstream stream(file_name.c_str());
		std::string line, field;
		while(stream){
			std::getline(stream, line);
			if(line == "") continue;
			elements_num ++;
			if(features_num == 0){
				std::stringstream line_stream(line);
				while(std::getline(line_stream, field, delimiter)){
					features_num ++;
				}
			}
		}

		std::cout<<"elements:"<<elements_num<<" features:"<<features_num<<std::endl;

		DBSCAN::ClusterData data( elements_num, features_num );
		data_min.resize( features_num, std::numeric_limits<double>::max() );
		data_max.resize( features_num, -std::numeric_limits<double>::max() );
		data_mid.resize( features_num );
		data_ranges.resize( features_num );

		stream.clear();
		stream.seekg(0, std::ios::beg);
		elements_num = 0; 
		double tmp = 0;
		while(stream){
			std::getline(stream, line);
			if(line == "") continue;
			features_num = 0;
			std::stringstream line_stream(line);
			while(std::getline(line_stream, field, delimiter)){
				tmp = atof(field.c_str());
				data(elements_num, features_num) = tmp;
				data_mid[features_num] += tmp;
				if (data_min[features_num] > tmp) {
					data_min[features_num] = tmp;
				}
				if (data_max[features_num] < tmp) {
					data_max[features_num] = tmp;
				}
				features_num ++;
			}
			elements_num ++;
		}
		for (uint32_t i = 0; i < data_mid.size(); i++) {
			data_mid[i] /= elements_num;
			data_ranges[i] = data_max[i] == data_min[i] ? 1 : data_max[i] - data_min[i]; 		
		}
		std::cout<<"min data:"<<std::endl;
		std::copy(data_min.begin(), data_min.end(), std::ostream_iterator<double>(std::cout, " "));
		std::cout<<std::endl;
		std::cout<<"max data:"<<std::endl;
		std::copy(data_max.begin(), data_max.end(), std::ostream_iterator<double>(std::cout, " "));
		std::cout<<std::endl;
		std::cout<<"mid data:"<<std::endl;
		std::copy(data_mid.begin(), data_mid.end(), std::ostream_iterator<double>(std::cout, " "));
		std::cout<<std::endl;
		std::cout<<"data ranges:"<<std::endl;
		std::copy(data_ranges.begin(), data_ranges.end(), std::ostream_iterator<double>(std::cout, " "));
		std::cout<<std::endl;

		stream.close();
		std::cout<<"Reading finished"<<std::endl;
		return data;
	}

	DBSCAN::DBSCAN() {

	}

	DBSCAN::DBSCAN( double eps, size_t minPts, int num_threads )
	: m_eps( eps )
	, m_minPts( minPts )
	, m_num_threads( num_threads ) {
		reset();
	}

	DBSCAN::~DBSCAN() {

	}

	void DBSCAN::reset() {
		cluster_labels.clear();
	}

	void DBSCAN::init_labels( size_t s ) {
		cluster_labels.resize(s);
		for( auto & l : cluster_labels) {
			l = UNCLASSIFIED;
		}
	}
	
	void DBSCAN::normalize( DBSCAN::ClusterData & C, std::vector<uint32_t> & selected_features ) {
		size_t row = C.size1();
		size_t col = C.size2();
		omp_set_dynamic(0); 
		omp_set_num_threads( m_num_threads );
		#pragma omp parallel for
		for (uint32_t i = 0; i < row; i++) {
			for (uint32_t j = 0; j < col; j++) {
				C(i, j) = C(i, j) - data_mid[j];
				if (selected_features[j] == 1) {
					C(i, j) = C(i, j) / data_ranges[j];
				}
			}
		}
	}

	DBSCAN::Neighbors DBSCAN::find_neighbors( DBSCAN::ClusterData & C, uint32_t pts, std::vector<double> & weights ) {
		size_t row = C.size1();
		std::vector<double> v_dist(row);
		Neighbors ne;

		omp_set_dynamic(0); 
		omp_set_num_threads( m_num_threads );
		#pragma omp parallel for
		for (uint32_t i = 0; i < row; i++) {
			if (i != pts) {
				ublas::matrix_row<DBSCAN::ClusterData> U (C, i);
				ublas::matrix_row<DBSCAN::ClusterData> V (C, pts);
				auto D_UV = U - V;
				for (uint32_t k = 0; k < D_UV.size(); k++) {
					v_dist[i] = v_dist[i] + weights[k]*D_UV(k)*D_UV(k);
				}
			}
		}

		for (uint32_t j = 0; j < row; j++) {
			if (v_dist[j] <= m_eps && j != pts) {
				ne.push_back(j);
			}
		}
		return ne;
	}

	void DBSCAN::dbscan( DBSCAN::ClusterData & C, std::vector<double> & weights ) {
		size_t row = C.size1();
		init_labels( row );
		uint32_t cluster_id = 0;
		Neighbors ne;
		for (uint32_t p = 0; p < row; p++) {
			if ( cluster_labels[p] == UNCLASSIFIED ) {
				ne = find_neighbors( C, p , weights);
				if (ne.size() >= m_minPts) {
					cluster_labels[p] = cluster_id;
					for (const auto & n : ne) {
						cluster_labels[n] = cluster_id;
					}
					expand_cluster(C, ne, cluster_id, weights);
					cluster_id++;
				} else {
					cluster_labels[p] = NOISE;
				}
			}
		}
	}

	void DBSCAN::expand_cluster( DBSCAN::ClusterData & C, DBSCAN::Neighbors & ne, uint32_t cluster_id, std::vector<double> & weights ) {
		Neighbors ne1;
		uint32_t pts;
		for (uint32_t i = 0; i < ne.size(); i++) {
			pts = ne[i];
			ne1 = find_neighbors(C, pts, weights);
			if ( ne1.size() >= m_minPts ) {
				cluster_labels[pts] = cluster_id;
				for (const auto & n1 : ne1) {
					if ( cluster_labels[n1] == UNCLASSIFIED || cluster_labels[n1] == NOISE ) {
						if ( cluster_labels[n1] == UNCLASSIFIED ) {
							ne.push_back(n1);
						}
						cluster_labels[n1] = cluster_id;
					}
				}
			}
			ne.erase(ne.begin() + i);
			i--;
		}
	}

	const DBSCAN::Labels & DBSCAN::get_labels() const {
		return cluster_labels;
	}

	DBSCAN::ClusterMap DBSCAN::gen_cluster_map() {
		uint32_t row = cluster_labels.size();
		DBSCAN::ClusterMap cmap;
		DBSCAN::ClusterMap::iterator iter;
		for ( uint32_t i=0; i<row; i++ ) {
			iter = cmap.find(cluster_labels[i]);
			if (iter == cmap.end()) {
				std::vector<uint32_t> value(1, i);
				cmap.insert(std::make_pair(cluster_labels[i], value));
			} else {
				iter->second.push_back(i);
			}
		}
		return cmap;
	}

	void DBSCAN::print_cluster_stat(DBSCAN::ClusterMap & cmap) {
		DBSCAN::ClusterMap::iterator iter;
		for (iter=cmap.begin(); iter!=cmap.end(); iter++) {
			std::cout<<"key:"<<iter->first<<" value:"<<iter->second.size()<<std::endl;
		}
	}

	int32_t DBSCAN::get_max_cluster(DBSCAN::ClusterMap & cmap) {
		DBSCAN::ClusterMap::iterator iter;
		int32_t kmax = 0;
		uint32_t vmax = 0;
		for (iter=cmap.begin(); iter!=cmap.end(); iter++) {
			if (iter->second.size() > vmax) {
				vmax = iter->second.size();
				kmax = iter->first;
			}
		}
		std::cout<<"max cluster (key:"<<kmax<<" value:"<<vmax<<")"<<std::endl;
		return kmax;
	}

	void DBSCAN::write_max_cluster(int32_t id, std::string & file_name, std::ostream & o) {
		std::ifstream stream(file_name.c_str());
		std::string line;
		size_t cnt = 0;
		while(stream){
			std::getline(stream, line);
			if(line == "") continue;
			if(cluster_labels[cnt] == id){
				o << line << std::endl;
			}
			cnt ++;
		}
	}

	std::ostream& operator<<( std::ostream& o, DBSCAN & d ) {
		for ( const auto & l : d.get_labels() ) {
			o << l << std::endl;
		}
		return o;
	}
}
