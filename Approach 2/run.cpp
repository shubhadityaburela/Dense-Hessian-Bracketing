#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <numeric>
#include <random>
#include <chrono>
#include "eigen/Eigen/Dense"



template<typename T>
struct Triplet {

	T x; T y; T z;

	// constructor
	Triplet()
		: x(0), y(0), z(0)
	{}

	// destructor
	~Triplet() {};
};


template<typename T, typename S>
struct Bracket_Triplet {

	T x; T y; T z;
	S unique_id;

	// constructor
	Bracket_Triplet()
		: x(0), y(0), z(0)
	{}

	// destructor
	~Bracket_Triplet() {};
};


template<typename T>
struct Cost_Triplet {

	T cost;
	T split_pos;
	Triplet<T> dim;

	// constructor
	Cost_Triplet()
		: cost(0), split_pos(0)
	{}

	// destructor
	~Cost_Triplet() {};

};


template<typename T>
class Tensor {
public:

	std::vector<std::vector<std::vector<float>>> data;
	Tensor() {};

	// Randomize the elements of Tensor
	void Random(int x, int y, int z) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(1.0, 4.0);

		data.resize(x);
		for (int i = 0; i < x; i++) {
			data[i].resize(y);
			for (int j = 0; j < y; j++) {
				data[i][j].resize(z);
				for (int k = 0; k < z; k++) {
					data[i][j][k] = dis(gen);
				}
			}
		}
	}

	// Initialize the Tensors
	void init(int x, int y, int z) {
		data.resize(x);
		for (int i = 0; i < x; i++) {
			data[i].resize(y);
			for (int j = 0; j < y; j++) {
				data[i][j].resize(z);
			}
		}
	}

	// Assignment operator for Tensors 
	void operator=(const Tensor<T>& othertensor) {
		this->init(othertensor.data.size(), othertensor.data[0].size(), othertensor.data[0][0].size());
		for (int i = 0; i < othertensor.data.size(); i++) {
			for (int j = 0; j < othertensor.data[i].size(); j++) {
				for (int k = 0; k < othertensor.data[i][j].size(); k++) {
					this->data[i][j][k] = othertensor.data[i][j][k];
				}
			}
		}
	}

	// Add two Tensors
	Tensor<T> operator+(const Tensor<T>& othertensor) {
		Tensor<T> result;
		result.data.resize(this->data.size());
		for (int i = 0; i < othertensor.data.size(); i++) {
			result.data[i].resize(this->data[i].size());
			for (int j = 0; j < othertensor.data[i].size(); j++) {
				result.data[i][j].resize(this->data[i][j].size());
				for (int k = 0; k < othertensor.data[i][j].size(); k++) {
					result.data[i][j][k] = this->data[i][j][k] + othertensor.data[i][j][k];
				}
			}
		}
		return result;
	}

	~Tensor() {};
};


// Matrix-Tensor Product
template<typename T>
Tensor<T> MatXTens(const Eigen::MatrixXf& A, const Tensor<T>& B) {

	Tensor<T> result;
	result.init(A.rows(), B.data[0].size(), B.data[0][0].size());

	for (int k = 0; k < B.data[0][0].size(); k++) {
		for (int i = 0; i < A.rows(); i++) {
			for (int j = 0; j < B.data[0].size(); j++) {
				for (int s = 0; s < A.cols(); s++) {
					result.data[i][j][k] += A(i, s) * B.data[s][j][k];
				}
			}
		}
	}
	return result;
}


// Tensor-Matrix Product
template<typename T>
Tensor<T> TensXMat(const Tensor<T>& B, const Eigen::MatrixXf& A) {

	Tensor<T> result;
	result.init(B.data.size(), B.data[0].size(), A.cols());

	for (int j = 0; j < B.data[0].size(); j++) {
		for (int i = 0; i < B.data.size(); i++) {
			for (int k = 0; k < A.cols(); k++) {
				for (int s = 0; s < B.data[0][0].size(); s++) {
					result.data[i][j][k] += B.data[i][j][s] * A(s, k);
				}
			}
		}
	}
	return result;
}


// Tensor-Matrix dyadic product
template<typename T>
Tensor<T> TensXMatCross(const Tensor<T>& B, const Eigen::MatrixXf& A) {

	Tensor<T> result;
	result.init(B.data.size(), B.data[0][0].size(), A.cols());

	for (int k = 0; k < B.data[0][0].size(); k++) {
		for (int i = 0; i < B.data.size(); i++) {
			for (int j = 0; j < A.cols(); j++) {
				for (int s = 0; s < B.data[0].size(); s++) {
					result.data[i][k][j] += B.data[i][s][k] * A(s, j);
				}
			}
		}
	}
	return result;
}


template<typename T, typename S>
void Hessian_Product_evaluation(std::vector<std::vector<Eigen::MatrixXf>>& Jacobian_products,
	const std::vector<Eigen::MatrixXf>& Jacobian_factors,
	std::vector<std::vector<Tensor<T>>>& Hessian_products,
	const std::vector<Tensor<T>>& Hessian_factors,
	const std::vector<std::vector<Bracket_Triplet<T, S>>>& B) {

	Tensor<T> Hessian;

	int p = Hessian_factors.size();
	for (int i = 0; i < p; i++) {
		Jacobian_products[i][i] = Jacobian_factors[i];
		Hessian_products[i][i] = Hessian_factors[i];
	}

	int d = 0;
	int cnt_col = 0;
	int cnt_row = 0;
	for (auto& itrx : B) {
		for (auto& itry : itrx) {
			if ((itry.x < d && itry.z < d) || (itry.x > d && itry.z > d)) {
				Jacobian_products[itry.x][itry.z] = Jacobian_products[itry.x][itry.y] *
					Jacobian_products[itry.y - 1][itry.z];
			}
			if (itry.x >= d && itry.z <= d) {
				if (itry.y <= d) {
					Hessian_products[itry.x][itry.z] = TensXMat(Hessian_products[itry.x][itry.y],
						Jacobian_products[itry.y - 1][itry.z]);
				}
				if (itry.y > d) {
					Hessian_products[itry.x][itry.z] = MatXTens(Jacobian_products[itry.x][itry.y],
						Hessian_products[itry.y - 1][itry.z]);
				}
			}
			cnt_col++;
		}
		if (d == 0) { Hessian = Hessian_products[p - 1][0]; }
		else { Hessian = Hessian + TensXMatCross(Hessian_products[p - 1][0], Jacobian_products[d - 1][0]); }
		d++;
		cnt_row++;
	}
}


template<typename T>
void Bracket_left(const std::vector<Eigen::MatrixXf>& Jacobian_factors,
	const std::vector<Tensor<T>>& Hessian_factors,
	const int d) {

	T p = Jacobian_factors.size();
	Tensor<T> Hess_prod;
	Eigen::MatrixXf Jac_prod;

	// Bracketing from left ((((A)B)C)D.....X)
	Jac_prod = Jacobian_factors[p - 1];
	Hess_prod = Hessian_factors[p - 1];
	for (int j = p - 1; j > d + 1; j--) { Jac_prod = Jac_prod * Jacobian_factors[j - 1]; }
	if (d < p - 1) { Hess_prod = MatXTens(Jac_prod, Hessian_factors[d]); }
	for (int j = d - 1; j >= 0; j--) { Hess_prod = TensXMat(Hess_prod, Jacobian_factors[j]); }
	if (d > 0) {
		Jac_prod = Jacobian_factors[d - 1];
		for (int j = d - 2; j >= 0; j--) {
			Jac_prod = Jac_prod * Jacobian_factors[j];
		}
		Hess_prod = TensXMatCross(Hess_prod, Jac_prod);
	}

}


template<typename T>
void Bracket_right(const std::vector<Eigen::MatrixXf>& Jacobian_factors,
	const std::vector<Tensor<T>>& Hessian_factors,
	const int d) {

	T p = Jacobian_factors.size();
	Tensor<T> Hess_prod;
	Eigen::MatrixXf Jac_prod;


	// Bracketing from right (A(B(C(D.....(X)))))
	Jac_prod = Jacobian_factors[0];
	Hess_prod = Hessian_factors[0];
	for (int j = 1; j < d; j++) { Jac_prod = Jacobian_factors[j] * Jac_prod; }
	if (d > 0) { Hess_prod = TensXMat(Hessian_factors[d], Jac_prod); }
	for (int j = d + 1; j < p; j++) { Hess_prod = MatXTens(Jacobian_factors[j], Hess_prod); }
	if (d > 0) { Hess_prod = TensXMatCross(Hess_prod, Jac_prod); }

}



int main(int argc, char* argv[]) {

	assert(argc == 3); std::ifstream in_1(argv[1]); std::ifstream in_2(argv[2]);
	using T = unsigned long;
	using S = std::string;


	T p; in_1 >> p; assert(p > 0);                    // This gives us the order of compositeness of the hessian 

	std::vector<Triplet<T>> Jac(p);                   // Jacobian vector
	std::vector<Triplet<T>> Hess(p);                  // Hessian vector
	std::vector<std::vector<Bracket_Triplet<T, S>>>
		B(p, std::vector<Bracket_Triplet<T, S>>(p - 1));      // Stores the Bracketing order for the individual chains

	// Input vector for Jacobian and Hessian size
	for (unsigned int i = 0; i < p; i++) {
		in_1 >> Jac[i].x >> Jac[i].y;
		Hess[i].x = Jac[i].x;
		Hess[i].y = Jac[i].y;
		Hess[i].z = Jac[i].y;
		Jac[i].z = 1;
	}

	// Input vector for optimal Bracketing order of the Hessian
	for (unsigned int i = 0; i < p; i++) {
		for (unsigned int j = 0; j < (p - 1); j++) {
			in_2 >> B[i][j].x >> B[i][j].y >> B[i][j].z >> B[i][j].unique_id;
		}
	}


	std::vector<Eigen::MatrixXf> Jacobian_factors;
	std::vector<Tensor<T>> Hessian_factors(p);
	std::vector<std::vector<Eigen::MatrixXf>> Jacobian_products(p, std::vector<Eigen::MatrixXf>(p));
	std::vector<std::vector<Tensor<T>>> Hessian_products(p, std::vector<Tensor<T>>(p));
	for (unsigned int i = 0; i < p; i++) {
		Jacobian_factors.push_back(Eigen::MatrixXf::Random(Jac[i].x, Jac[i].y));
		Hessian_factors[i].Random(Hess[i].x, Hess[i].y, Hess[i].z);
	}


	std::cout << "Elapsed time (in microseconds):" << std::endl;
	
	// Brute force way of computing the FMA for the Hessian chain 
	auto duration_left = 0;
	auto duration_right = 0;
	for (unsigned int k = 0; k < p; k++) {
		// Bracket from left
		auto t1 = std::chrono::high_resolution_clock::now();
		Bracket_left(Jacobian_factors, Hessian_factors, k);
		auto t2 = std::chrono::high_resolution_clock::now();
		auto duration_1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		duration_left = duration_left + duration_1;

		// Bracket from right
		auto t3 = std::chrono::high_resolution_clock::now();
		Bracket_right(Jacobian_factors, Hessian_factors, k);
		auto t4 = std::chrono::high_resolution_clock::now();
		auto duration_2 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
		duration_right = duration_right + duration_2;
	}
	
	// Dynamic Programming Algorithm
	auto t5 = std::chrono::high_resolution_clock::now();
	Hessian_Product_evaluation(Jacobian_products, Jacobian_factors,
		Hessian_products, Hessian_factors, B);
	auto t6 = std::chrono::high_resolution_clock::now();
	auto duration_opt = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();


	std::cout << "left bracketing: " << duration_left << std::endl;
	std::cout << "right bracketing: " << duration_right << std::endl;
	std::cout << "optimized bracketing: " << duration_opt << std::endl;


	return 0;
}
