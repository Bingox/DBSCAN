#include <vector>
#include <map>
#include <string>
#include <boost/numeric/ublas/matrix.hpp>

using namespace boost::numeric;

namespace clustering {
	class DBSCAN {

	public:
		typedef ublas::matrix<double> ClusterData;
		typedef std::vector<uint32_t> Neighbors;
		typedef std::vector<int32_t> Labels;
		typedef std::map<int32_t, std::vector<uint32_t>> ClusterMap;

		DBSCAN(double eps, size_t minPts, int num_threads=1);
		DBSCAN();
		~DBSCAN();

		void reset();
		ClusterData read_file( std::string &file_name, char delimiter );
		void normalize( ClusterData & C, std::vector<uint32_t> & selected_features );
		void dbscan( ClusterData & C, std::vector<double> & weights );
		const Labels & get_labels() const;
		ClusterMap gen_cluster_map();
		void print_cluster_stat(ClusterMap & cmap);
		int32_t get_max_cluster(ClusterMap & cmap);
		void write_max_cluster(int32_t id, std::string & file_name, std::ostream & o);
	
	private:
		void init_labels( size_t s );
		Neighbors find_neighbors( ClusterData & C, uint32_t pts, std::vector<double> & weights );
		void expand_cluster( ClusterData & C, Neighbors & ne, uint32_t cluster_id, std::vector<double> & weights );
		double m_eps;
		size_t m_minPts;
		int m_num_threads;

		std::vector<double> data_min;
		std::vector<double> data_max;
		std::vector<double> data_mid;
		std::vector<double> data_ranges;

		const int32_t UNCLASSIFIED = -1;
		const int32_t NOISE = -2;

		Labels cluster_labels;
	};

	std::ostream& operator<<(std::ostream& o, DBSCAN & d);
}
