#include "metis.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

vector<idx_t> func(vector<idx_t> xadj, vector<idx_t> adjncy, vector<idx_t> vwgt, 
	idx_t parts, idx_t *options) {
	idx_t nVertices = xadj.size() - 1;
	idx_t nEdges = adjncy.size() / 2;
	idx_t nWeights = 1;
	idx_t nParts = parts;
	idx_t objval;
	std::vector<idx_t> part(nVertices, 0);
	int ret = METIS_PartGraphKway(&nVertices, &nWeights, xadj.data(), adjncy.data(),
		vwgt.data(), NULL, NULL, &nParts, NULL,
		NULL, options, &objval, part.data());
	std::cout << ret << std::endl;
	for (unsigned part_i = 0; part_i < part.size(); part_i++) {
		std::cout << part_i << " " << part[part_i] << std::endl;
	}
	return part;
}

int main() {
	string file = "graph.txt"; // weighted graph to be partitioned

	ifstream ingraph(filepath+file);
	if (!ingraph) {
		cout << "Fail to open the input file." << endl;
		exit(1);
	}

	int vexnum, edgenum;
	string line;
	getline(ingraph, line);
	istringstream tmp(line);
	tmp >> vexnum >> edgenum;
	vector<idx_t> xadj(0);
	vector<idx_t> adjncy(0);
	vector<idx_t> vwgt(0);
	idx_t a, w;
	for (int i = 0; i < vexnum; i++) {
		xadj.push_back(adjncy.size());
		getline(ingraph, line);
		istringstream tmp(line);
		while (tmp >> a >> w) {
			adjncy.push_back(a);
			vwgt.push_back(w);
		}
	}
	xadj.push_back(adjncy.size());
	ingraph.close();

	idx_t options[METIS_NOPTIONS];
	METIS_SetDefaultOptions(options);
	options[METIS_OPTION_CONTIG] = 1;
	options[METIS_OPTION_UFACTOR] = 5000; // u factor

	for (idx_t parts = 5; parts <= 50; parts += 5)
	{
		cout << parts << endl;

		vector<idx_t> part = func(xadj, adjncy, vwgt, parts, options);

		ofstream outpartition(filepath + string("part") + to_string(parts)
			+string("_u") + to_string(options[METIS_OPTION_UFACTOR])+string("-")+file);
		if (!outpartition) {
			cout << "Fail to open the input file." << endl;
			exit(1);
		}

		for (int i = 0; i < part.size(); i++) {
			outpartition << i << " " << part[i] << endl;
		}
		outpartition.close();
	}
}