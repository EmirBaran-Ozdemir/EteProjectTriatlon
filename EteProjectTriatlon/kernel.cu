
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>
#include <curand_kernel.h>
#include <random>
#include <chrono>
#include <thread>
#include <iostream>
#include <map>


struct Player
{
	int velocity;
	int location;
	bool willDisplay;
	int playerIdleTime;
};

const int swimPartLength = 5;
const int bicyclePartLength = 5;
const int runPartLength = 5;

const int delayBetweenParts = 10;

__global__ void calculatePlayersLocation(Player* player, bool* status, int* order, int* currentIndex)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	Player* currPlayer = &player[tid];
	bool* currStatus = &status[tid];
	if (currPlayer->location > swimPartLength + bicyclePartLength + runPartLength)
	{
		return;
	}
	if (currPlayer->playerIdleTime > 0)
	{
		currPlayer->playerIdleTime--;
		return;
	}
	int currentVelocity = currPlayer->velocity;

	if (currPlayer->location < swimPartLength + bicyclePartLength && currPlayer->location > swimPartLength)
	{
		currPlayer->playerIdleTime = 10;
		currentVelocity = currPlayer->velocity * 3;
	}
	else if (currPlayer->location > swimPartLength + bicyclePartLength)
	{
		currPlayer->playerIdleTime = 10;
		currentVelocity = std::ceil(float(currPlayer->velocity) / 3.f);
	}

	currPlayer->location += currentVelocity;
	if (currPlayer->location > swimPartLength + bicyclePartLength + runPartLength)
	{
		int index = atomicAdd(currentIndex, 1);
		int* currOrder = &order[index];
		*currOrder = tid;
		*currStatus = true;
	}
	if (currPlayer->willDisplay)
		printf("Player %d location: %d\n", tid, currPlayer->location);
}

void initializePlayers(Player* players, std::uniform_int_distribution<std::mt19937::result_type>& dist, std::mt19937& rng, int playerCount, std::vector<std::map<int, int>> playersToDisplay, std::vector<int> groupsToDisplay)
{
	for (int i = 0; i < playerCount; i++)
	{
		players[i].velocity = dist(rng);
		players[i].location = 0;
		players[i].willDisplay = false;
		players[i].playerIdleTime = 0;

		if (std::any_of(groupsToDisplay.begin(), groupsToDisplay.end(), [i](int group) { return i / 3 == group - 1; }))
		{
			players[i].willDisplay = true;
			players[i + 1].willDisplay = true;
			players[i + 1].velocity = dist(rng);
			players[i + 1].location = 0;
			players[i + 1].playerIdleTime = 0;
			players[i + 2].willDisplay = true;
			players[i + 2].velocity = dist(rng);
			players[i + 2].location = 0;
			players[i + 2].playerIdleTime = 0;
			i = i + 2;
			std::cout << "Group " << i / 3 << " will be displayed" << std::endl;
		}
		else if (std::any_of(playersToDisplay.begin(), playersToDisplay.end(), [i](std::map<int, int>& player) { return player.find(i) != player.end(); }))
		{
			players[i].willDisplay = true;
			for (const auto& player : playersToDisplay) {
				if (player.find(i) != player.end()) {
					int groupNumber = player.at(i);
					std::cout << "Player " << i << " from group " << groupNumber << " will be displayed" << std::endl;
				}
			}
		}
	}
}

int main(int argc, char* argv[])
{
	std::vector<std::map<int, int>> playersToDisplay;
	std::vector<int> groupsToDisplay;

	if (argc > 1) {
		for (int i = 1; i < argc; i++) {
			int tempPlayer = -1;
			int tempGroup = -1;
			if (std::strcmp(argv[i], "-P") == 0) {
				if (i + 1 < argc) {
					std::string playerArg(argv[i + 1]);
					size_t dashPos = playerArg.find('-');
					if (dashPos == std::string::npos || dashPos == 0 || dashPos == playerArg.length() - 1) {
						std::cerr << "Error: Invalid player format. Please use GROUP-INDEX format, e.g., 3-2" << std::endl;
						return 1;
					}
					std::string groupStr = playerArg.substr(0, dashPos);
					std::string indexStr = playerArg.substr(dashPos + 1);
					int group = std::stoi(groupStr);
					int index = std::stoi(indexStr);
					if (group <= 0 || group > 300 || index <= 0 || index > 3) {
						std::cerr << "Error: Invalid player or group number" << std::endl;
						return 1;
					}
					tempPlayer = index;
					tempGroup = group;
					i++;
				}
				else {
					std::cerr << "Error: Missing GROUP-INDEX after -P" << std::endl;
					return 1;
				}
			}
			else if (std::strcmp(argv[i], "-G") == 0) {

				if (i + 1 < argc) {
					if (std::atoi(argv[i + 1]) <= 0 || std::atoi(argv[i + 1]) > 300)
					{
						std::cerr << "Error: Invalid group number" << std::endl;
						return 1;
					}
					tempGroup = std::atoi(argv[i + 1]);
					i++;
				}
				else {
					std::cerr << "Error: Missing group number after -G" << std::endl;
					return 1;
				}
			}
			else {
				std::cerr << "Error: Invalid argument " << argv[i] << std::endl;
				return 1;
			}
			if (tempPlayer != -1 && tempGroup != -1) {
				playersToDisplay.emplace_back(std::map<int, int>{{tempPlayer, tempGroup}});
			}
			else if (tempGroup != -1)
			{
				groupsToDisplay.push_back(tempGroup);
			}
		}
	}
	else {
		std::cerr << "Usage: program.exe -G <group_number> -P <player_number>, -G <group_number> -P <player_number>, ..." << std::endl;
		std::cerr << "Group number should be between 0 and 300, player number should be between 1 and 3" << std::endl;
		return 1;
	}

	const int GROUP_SIZE = 3;
	const int GROUP_COUNT = 300;

	const int NUM_THREAD = GROUP_SIZE;
	const int NUM_BLOCKS = GROUP_COUNT;

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> dist5(1, 5);


	size_t bytesPlayer = NUM_THREAD * NUM_BLOCKS * sizeof(Player);
	size_t bytesBool = NUM_THREAD * NUM_BLOCKS * sizeof(bool);
	size_t bytesOrder = NUM_THREAD * NUM_BLOCKS * sizeof(int);
	Player* d_a;
	bool* d_b;
	int* d_order;
	int* d_currentIndex;

	bool* h_b = (bool*)malloc(bytesBool);
	int* h_order = (int*)malloc(bytesOrder);
	memset(h_b, 0, bytesBool);
	memset(h_order, 0, bytesOrder);
	Player* h_a = (Player*)malloc(bytesPlayer);

	initializePlayers(h_a, dist5, rng, GROUP_SIZE * GROUP_COUNT, playersToDisplay, groupsToDisplay);

	cudaMalloc(&d_a, bytesPlayer);
	cudaMalloc(&d_b, bytesBool);
	cudaMalloc(&d_order, bytesOrder);
	cudaMalloc(&d_currentIndex, sizeof(int));

	cudaMemcpy(d_a, h_a, bytesPlayer, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytesBool, cudaMemcpyHostToDevice);
	cudaMemcpy(d_order, h_order, bytesOrder, cudaMemcpyHostToDevice);
	bool displayedFirstOne = false;
	while (true)
	{
		calculatePlayersLocation << < NUM_BLOCKS, NUM_THREAD >> > (d_a, d_b, d_order, d_currentIndex);
		cudaMemcpy(h_b, d_b, bytesBool, cudaMemcpyDeviceToHost);
		bool anyFalse = false;
		bool anyTrue = std::any_of(h_b, h_b + GROUP_COUNT * GROUP_SIZE, [](bool val) { return val; });
		if (anyTrue && !displayedFirstOne)
		{
			cudaMemcpy(h_a, d_a, bytesPlayer, cudaMemcpyDeviceToHost);
			for(int i = 0; i < GROUP_COUNT * GROUP_SIZE; i++)
			{
				printf("Player %d location: %d\n", i, h_a[i].location);
			}
			displayedFirstOne = true;
		}
		bool allTrue = std::all_of(h_b, h_b + GROUP_COUNT * GROUP_SIZE, [](bool val) { return val; });
		if (allTrue)
			break;
		std::cout << "Not all players finished yet" << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	cudaMemcpy(h_order, d_order, bytesOrder, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_a, d_a, bytesPlayer, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_order);

	for(int i = 0; i < GROUP_COUNT * GROUP_SIZE; i++)
	{
		printf("Player %d finished at %d\n", h_order[i], i );
	}

	return 0;
}