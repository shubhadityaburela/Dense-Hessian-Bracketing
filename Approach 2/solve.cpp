
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <limits>
#include <numeric>
#include <cstdlib>
#include <string>
#include <sstream>


template<typename T, typename S>
struct Triplet {

	T x; T y; T z;
	S unique_id;

	// constructor
	Triplet()
		: x(0), y(0), z(0)
	{}

	// destructor
	~Triplet() {};
};


template<typename T, typename S>
struct Cost_Triplet {

	T cost;
	T split_pos;
	Triplet<T, S> dim;

	// constructor
	Cost_Triplet()
		: cost(0), split_pos(0)
	{}

	// destructor
	~Cost_Triplet() {};

};


template<typename T, typename S>
struct String_cost {
	T cost;
	S split_pos;

	// default constructor
	String_cost() = default;

	// destructor
	~String_cost() {};
};


template<typename T, typename S>
struct node {

	Triplet<T, S> info;
	node *left;
	node *right;

	// constructor
	node()
		:left(nullptr), right(nullptr)
	{}

	node(const int& info)
		:info(info), left(nullptr), right(nullptr)
	{}

	// member function
	int max_depth() const {
		const int left_depth = left ? left->max_depth() : 0;
		const int right_depth = right ? right->max_depth() : 0;
		return (left_depth > right_depth ? left_depth : right_depth) + 1;
	}

	// destructor
	~node() { delete left; delete right; }

};


template<typename T, typename S>
class BST {

	node<T, S>* root;

public:

	// constructor
	BST()
	{
		root = NULL;
	}

	// member functions
	int get_max_depth() const { return root ? root->max_depth() : 0; }
	node<T, S>* insert(node<T, S> *, node<T, S> *);
	void inorder(node<T, S> *, std::vector<Triplet<T, S>>&);
	void postorder(node<T, S> *);
	void givenlevel(node<T, S> *, int, std::vector<Triplet<T, S>>&);
	void levelorder(node<T, S> *, std::vector<Triplet<T, S>>&);
	void display(node<T, S> *, int);
	void DeletefromBST(node<T, S> * &);
	void DestroyRecursive(node<T, S> *);
	void clear() { delete root; root = nullptr; }

	// destructor
	~BST() { delete root; }

private:

	// data members
	std::vector<Triplet<T, S>> node_collector;

};


template<typename T, typename S>
node<T, S>* BST<T, S>::insert(node<T, S> *tree, node<T, S> *newnode) {
	if (root == NULL)
	{
		root = new node<T, S>;
		root->info = newnode->info;
		root->left = NULL;
		root->right = NULL;

		return root;
	}
	if (tree->info.y == newnode->info.y)
	{
		std::cout << "Element already in the tree" << std::endl;
		return tree;
	}
	if (tree->info.y > newnode->info.y)
	{
		if (tree->right != NULL) {
			insert(tree->right, newnode);
		}
		else {
			tree->right = newnode;
			(tree->right)->left = NULL;
			(tree->right)->right = NULL;

			return tree;
		}
	}
	else {
		if (tree->left != NULL) {
			insert(tree->left, newnode);
		}
		else {
			tree->left = newnode;
			(tree->left)->left = NULL;
			(tree->left)->right = NULL;

			return tree;
		}

	}

	return tree;
}


template<typename T, typename S>
void BST<T, S>::inorder(node<T, S> *ptr, std::vector<Triplet<T, S>>& node_collector) {
	if (root == NULL)
	{
		std::cout << "Tree is empty" << std::endl;
		return;
	}
	if (ptr != NULL)
	{
		inorder(ptr->left, node_collector);
		node_collector.push_back(ptr->info);
		inorder(ptr->right, node_collector);
	}
}


template<typename T, typename S>
void BST<T, S>::postorder(node<T, S> *ptr)
{
	if (ptr == NULL)
		return;

	// first recur on left subtree 
	postorder(ptr->left);

	// then recur on right subtree 
	postorder(ptr->right);

	// now deal with the node 
	std::cout << ptr->info.y << " ";
}


template<typename T, typename S>
void BST<T, S>::givenlevel(node<T, S> *ptr, int level, std::vector<Triplet<T, S>>& BO) {

	if (ptr == NULL)
		return;
	if (level == 1) {
		BO.push_back(ptr->info);
	}
	else if (level > 1)
	{
		givenlevel(ptr->left, level - 1, BO);
		givenlevel(ptr->right, level - 1, BO);
	}

}


template<typename T, typename S>
void BST<T, S>::levelorder(node<T, S> *ptr, std::vector<Triplet<T, S>>& BO) {

	int h = BST::get_max_depth();
	for (int i = h; i >= 1; i--) {
		BST::givenlevel(ptr, i, BO);
	}

}


template<typename T, typename S>
void BST<T, S>::display(node<T, S> *ptr, int level) {
	int i;
	if (ptr != NULL)
	{
		display(ptr->right, level + 1);
		std::cout << std::endl;
		if (ptr == root) {
			std::cout << "Root->: ";
		}
		else {
			for (i = 0; i < level; i++) {
				std::cout << "     ";
			}
		}
		std::cout << "(" << ptr->info.x << ", " << ptr->info.y << ", " << ptr->info.z << ")   ";
		display(ptr->left, level + 1);
	}
}

template<typename T, typename S>
void BST<T, S>::DeletefromBST(node<T, S>* &ptr)
{

	if (ptr == NULL) { return; }
	DeletefromBST(ptr->left);
	DeletefromBST(ptr->right);

	std::cout << "\n Deleting node: " "(" << ptr->info.x << ", "
		<< ptr->info.y << ", " << ptr->info.z << ")   " << std::endl;

	delete ptr;

}

template<typename T, typename S>
void BST<T, S>::DestroyRecursive(node<T, S> * ptr)
{
	if (ptr) {
		DestroyRecursive(ptr->left);
		DestroyRecursive(ptr->right);
		delete ptr;
	}
}



template<typename T, typename S>
T dp(const std::vector<Triplet<T, S>> &B,
	std::vector<std::vector<Cost_Triplet<T, S>>> &C,
	const std::unordered_map<S, String_cost<T, S>> &saved_cost,
	const unsigned int cnt) {

	int p = B.size();

	for (int j = 0; j < p; j++) {
		for (int i = j; i >= 0; i--) {
			if (i == j) {
				C[j][i].cost = 0;
				C[j][i].split_pos = 0;
				C[j][i].dim = B[j];
			}
			else {
				S accumulation;
				for (int r = j; r >= i; r--)
					accumulation = accumulation + B[r].unique_id + ",";
				auto search = saved_cost.find(accumulation);

				if (search != saved_cost.end()) {
					C[j][i].cost = 0;// Set the cost of already accumulated subproblem as 0
					auto pos = search->second.split_pos;
					auto itr = std::find_if(B.begin(), B.end(), [pos](const auto& element)
						{return element.unique_id == pos; });
					C[j][i].split_pos = std::distance(B.begin(), itr);
				}
				else {
					if (j < cnt) {
						for (int k = i + 1; k <= j; k++) {
							T fma = B[j].x * B[k - 1].x * B[i].y;
							T cost = C[j][k].cost + C[k - 1][i].cost + fma;
							if (k == i + 1 || cost < C[j][i].cost) {
								C[j][i].cost = cost;
								C[j][i].split_pos = k;
							}
						}
						C[j][i].dim.x = B[j].x;
						C[j][i].dim.y = B[i].y;
						C[j][i].dim.z = 1;
					}
					else if (j == cnt) {
						for (int k = i + 1; k <= j; k++) {
							T fma = B[j].x * B[j].y * B[k - 1].x * B[i].y;
							T cost = C[j][k].cost + C[k - 1][i].cost + fma;
							if (k == i + 1 || cost < C[j][i].cost) {
								C[j][i].cost = cost;
								C[j][i].split_pos = k;
							}
						}
						C[j][i].dim.x = B[j].x;
						C[j][i].dim.y = B[j].y;
						C[j][i].dim.z = B[i].y;
					}
					else {
						if (i > cnt) {
							for (int k = i + 1; k <= j; k++) {
								T fma = B[j].x * B[k - 1].x * B[i].y;
								T cost = C[j][k].cost + C[k - 1][i].cost + fma;
								if (k == i + 1 || cost < C[j][i].cost) {
									C[j][i].cost = cost;
									C[j][i].split_pos = k;
								}
							}
							C[j][i].dim.x = B[j].x;
							C[j][i].dim.y = B[i].y;
							C[j][i].dim.z = 1;
						}
						else if (i == cnt) {
							for (int k = i + 1; k <= j; k++) {
								T fma = B[j].x * B[k - 1].x * B[i].y * B[i].z;
								T cost = C[j][k].cost + C[k - 1][i].cost + fma;
								if (k == i + 1 || cost < C[j][i].cost) {
									C[j][i].cost = cost;
									C[j][i].split_pos = k;
								}
							}
							C[j][i].dim.x = B[j].x;
							C[j][i].dim.y = B[i].y;
							C[j][i].dim.z = B[i].z;
						}
						else {
							for (int k = i + 1; k <= j; k++) {
								T fma = B[j].x * B[cnt].y * B[k].y * B[i].y;
								T cost = C[j][k].cost + C[k - 1][i].cost + fma;
								if (k == i + 1 || cost < C[j][i].cost) {
									C[j][i].cost = cost;
									C[j][i].split_pos = k;
								}
							}
							C[j][i].dim.x = B[j].x;
							C[j][i].dim.y = B[cnt].y;
							C[j][i].dim.z = B[i].y;
						}
					}
				}
			}
		}
	}
	return C[p - 1][0].cost;
}



template<typename T, typename S>
std::pair<T, T> Random_Bracket(const std::vector<Triplet<T, S>>& B,
	const int d) {

	T p = B.size();
	T fma_right = 0;
	T fma_left = 0;

	// Bracketing from left ((((A)B)C)D.....X)
	for (int j = p - 1; j > d + 1; j--) {
		fma_left = fma_left + B[p - 1].x * B[j - 1].x * B[j - 1].y;
	}
	if (d < p - 1) {
		fma_left = fma_left + B[p - 1].x * B[d].x * B[d].y * B[d].z;
	}
	for (int j = d - 1; j >= 0; j--) {
		fma_left = fma_left + B[p - 1].x * B[d].y * B[j].x * B[j].y;
	}
	if (d > 0) {
		fma_left = fma_left + B[p - 1].x * B[d].y * B[0].y * B[0].y;
	}


	// Bracketing from right (A(B(C(D.....(X)))))
	for (int j = 1; j < d; j++) {
		fma_right = fma_right + B[j].x * B[j - 1].x * B[0].y;
	}
	if (d > 0) {
		fma_right = fma_right + B[d].x * B[d].y * B[d - 1].x * B[0].y;
	}
	for (int j = d + 1; j < p; j++) {
		fma_right = fma_right + B[j].x * B[j].y * B[d].y * B[0].y;
	}
	if (d > 0) {
		fma_right = fma_right + B[p - 1].x * B[d].y * B[0].y * B[0].y;
	}

	return std::make_pair(fma_left, fma_right);

}




template<typename T, typename S>
void Bracketing_structure(const std::vector<std::vector<Cost_Triplet<T, S>>>& C,
	T n, BST<T, S> &bst, node<T, S> *root) {

	T start = 0;
	T end = 0;

	std::vector<Triplet<T, S>> Bin_vector;
	std::vector<T> diff;

	bst.inorder(root, Bin_vector);


	for (unsigned int i = 0; i <= Bin_vector.size(); i++) {

		if (i == 0)  start = n - 1;
		else start = end - 1;

		if (i == Bin_vector.size()) end = 0;
		else end = Bin_vector[i].y;

		node<T, S> *k;
		k = new node<T, S>;
		k->info.x = start;
		k->info.y = C[start][end].split_pos;
		k->info.z = end;


		if (k->info.y != 0) { bst.insert(root, k); }

		diff.push_back(start - end);

	}

	if (std::all_of(diff.begin(), diff.end(), [&](int p)
	{return p == 0; })) {
		return;
	}
	else { Bracketing_structure(C, n, bst, root); }
}






int main(int argc, char* argv[]) {

	assert(argc == 2); std::ifstream in(argv[1]);
	using T = unsigned long;
	using S = std::string;


	T p; in >> p; assert(p > 0);        // This gives us the order of compositeness of the hessian 

	std::vector<Triplet<T, S>> Jac(p);  // Vector of Jacobians
	std::vector<Triplet<T, S>> Hess(p); // Vector of Hessians 
	std::vector<Triplet<T, S>> B;       // Vector of Individual factors of the respective chains 
	std::vector<Triplet<T, S>> BO;      // Vector of the bracketing order

	for (unsigned int i = 0; i < p; i++) {
		in >> Jac[i].x >> Jac[i].y;
		Hess[i].x = Jac[i].x;
		Hess[i].y = Jac[i].y;
		Hess[i].z = Jac[i].y;
		Jac[i].z = 1;
		Jac[i].unique_id = std::to_string(i);
		Hess[i].unique_id = std::to_string(i + p);
	}

	std::unordered_map<S, String_cost<T, S>> saved_cost;   // costs saved for all the subproblems

	std::ofstream solution;
	solution.open("solution.txt");

	T term_cost = 0;
	T total_term_cost = 0;
	T Global_cost = 0;
	T left_cost = 0;
	T right_cost = 0;
	for (unsigned int k = 0; k < p; k++) {
		std::vector<std::vector<Cost_Triplet<T, S>>> C(p, std::vector<Cost_Triplet<T, S>>(p));

		// Individual chain formation by multiplying Jacobians and Hessians based on the formulation
		for (unsigned int j = 0; j < k; j++) { B.push_back(Jac[j]); }
		B.push_back(Hess[k]);
		for (unsigned int j = k + 1; j < p; j++) { B.push_back(Jac[j]); }

		term_cost = dp(B, C, saved_cost, k);

		std::cout << "Dynamic Programming Table for chain:" << k + 1 << std::endl;
		for (unsigned int j = 0; j < p; j++) {
			for (int i = j - 1; i >= 0; i--) {
				std::cout << "Cost(C[" << j << "][" << i << "])="
					<< C[j][i].cost << "; "
					<< "Split before G[" << C[j][i].split_pos << "]; " << std::endl;
			}
		}
		std::cout << std::endl;

		// Save the result of dp into a bracketing tree form 
		T size = p;
		BST<T, S> bst_C;
		node<T, S> *temp_C;
		temp_C = new node<T, S>;
		temp_C->info.x = size - 1;
		temp_C->info.y = C[size - 1][0].split_pos;
		temp_C->info.z = 0;
		node<T, S> *root = NULL;
		root = bst_C.insert(root, temp_C);
		Bracketing_structure(C, size, bst_C, root);
		bst_C.levelorder(root, BO);
		for (auto itr : BO) {
			S accumulation;
			for (auto r = itr.x; r >= itr.z && r <= itr.x;) {
				accumulation = accumulation + B[r].unique_id + ",";
				r--;
			}
			String_cost<T, S> state;
			state.cost = C[itr.x][itr.z].cost;
			state.split_pos = B[C[itr.x][itr.z].split_pos].unique_id;
			saved_cost.insert(std::make_pair(accumulation, state));
			solution << itr.x << " " << itr.y << " " << itr.z << " " << accumulation << std::endl;
		}
		solution << std::endl;

		S subch;
		if (k != 0) {
			for (auto r = k - 1; r >= 0 && r <= k - 1;) {
				subch = subch + B[r].unique_id + ",";
				r--;
			}
			auto search = saved_cost.find(subch);
			if (search != saved_cost.end()) {
				total_term_cost = term_cost + B[p - 1].x * B[k].y * B[0].y * B[0].y;
			}
			else {
				total_term_cost = term_cost + C[k - 1][0].cost +
					B[p - 1].x * B[k].y * B[0].y * B[0].y;
			}
		}
		else {
			total_term_cost = term_cost;
		}
		Global_cost = Global_cost + total_term_cost;


		// Brute force way of computing the FMA for the Hessian chain 
		// (Bracket from left and right)
		std::pair<T, T> Bracket_cost = Random_Bracket(B, k);
		left_cost = left_cost + Bracket_cost.first;
		right_cost = right_cost + Bracket_cost.second;

		BO.clear();
		B.clear();
	}
	solution.close();

	std::cout << "left bracketing fma = " << left_cost << std::endl;
	std::cout << "right bracketing fma = " << right_cost << std::endl;
	std::cout << "optimized bracketing fma = " << Global_cost << std::endl;

	return 0;
}
