#include <vector>
#include <string>
#include <boost/numeric/ublas/matrix.hpp>

using namespace boost::numeric;

namespace clustering {
	class DBSCAN {

	public:
		typedef ublas::matrix<double> ClusterData;
		typedef std::vector<uint32_t> Neighbors;
		typedef std::vector<int32_t> Labels;

		DBSCAN(double eps, size_t minPts, int num_threads=1);
		DBSCAN();
		~DBSCAN();

		void reset();
		ClusterData read_file( std::string &file_name, char delimiter );
		void dbscan( ClusterData & C );
		const Labels & get_labels() const;
	
	private:
		void init_labels( size_t s );
		Neighbors find_neighbors( ClusterData & C, uint32_t pts);
		void expand_cluster( ClusterData & C, Neighbors & ne, uint32_t cluster_id );
		double m_eps;
		size_t m_minPts;
		int m_num_threads;

		const int32_t UNCLASSIFIED = -1;
		const int32_t NOISE = -2;

		Labels cluster_labels;
	};

	std::ostream& operator<<(std::ostream& o, DBSCAN & d);
}
