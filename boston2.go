package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/sajari/regression"
)

func main() {
	start := time.Now() // Start measuring CPU time

	// Read CSV

	file, err := os.Open("housing1.csv")
	if err != nil {
		log.Fatalf("failed to open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Skip the header row
	_, err = reader.Read()
	if err != nil {
		log.Fatalf("failed to read header: %v", err)
	}

	records, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("failed to read CSV: %v", err)
	}

	// Check if any records were read
	if len(records) == 0 {
		log.Fatalf("no data in the CSV file")
	}

	var data [][]float64
	for _, record := range records {
		var floats []float64
		// Start from index 1 to skip the first column (neighborhood)
		for _, value := range record[1:] {
			val, err := strconv.ParseFloat(value, 64)
			if err != nil {
				log.Fatalf("failed to parse float: %v", err)
			}
			floats = append(floats, val)
		}
		data = append(data, floats)
	}

	responseIndex := len(data[0]) - 1
	y := make([]float64, len(data))
	for i, row := range data {
		y[i] = row[responseIndex]
	}

	numExplanatory := len(data[0]) - 1

	// Channels for communicating results
	results := make(chan result)
	done := make(chan struct{})

	// Start goroutines for fitting models
	for size := 4; size <= numExplanatory; size++ {
		go func(size int) {
			defer func() { done <- struct{}{} }()

			localBestAIC := math.Inf(1)
			var localBestFeatures []int
			var localBestMSE float64

			combinations := generateCombinations(numExplanatory, size)
			for _, features := range combinations {
				mse, aic := fitModel(y, features, data)

				if aic < localBestAIC {
					localBestAIC = aic
					localBestFeatures = features
					localBestMSE = mse
				}
			}

			// Send the results back to the main goroutine
			results <- result{localBestFeatures, localBestAIC, localBestMSE}
		}(size)
	}

	// Wait for all goroutines to finish
	go func() {
		for i := 4; i <= numExplanatory; i++ {
			<-done
		}
		close(results) // Close the results channel after all goroutines finish
	}()

	// Process results from the channel
	for res := range results {
		fmt.Printf("Best Model Features: %v\n", res.Features)
		fmt.Printf("Best Model AIC: %.4f\n", res.AIC)
		fmt.Printf("Best Model MSE: %.4f\n", res.MSE)
	}

	elapsed := time.Since(start)
	fmt.Printf("CPU time taken: %s\n", elapsed)
}

type result struct {
	Features []int
	AIC      float64
	MSE      float64
}

func fitModel(y []float64, features []int, data [][]float64) (mse, aic float64) {
	var (
		xs [][]float64
		f  float64
		r  regression.Regression
	)

	// Set the observed variable
	r.SetObserved("mv")

	// Prepare the feature data and add the selected features to the regression model
	for _, idx := range features {
		varName := strconv.Itoa(idx)
		// Define the function to extract the feature and add it directly to the regression model
		r.SetVar(idx, varName)
		// Prepare the feature data
		var x []float64
		for _, row := range data {
			x = append(x, row[idx])
		}
		xs = append(xs, x)
	}

	// Train the regression model
	for i, row := range xs {
		r.Train(regression.DataPoint(y[i], row))
	}

	// Run the regression
	r.Run()

	// Calculate MSE
	for i, row := range xs {
		yPred, _ := r.Predict(row)
		f += math.Pow(y[i]-yPred, 2)
	}
	mse = f / float64(len(xs))

	// Calculate AIC
	aic = float64(len(xs))*math.Log(mse) + 2.0*float64(len(features))

	return mse, aic
}

func generateCombinations(n, k int) [][]int {
	var combinations [][]int
	generateCombinationsHelper(n, k, 0, []int{}, &combinations)
	return combinations
}

func generateCombinationsHelper(n, k, index int, combination []int, combinations *[][]int) {
	if k == 0 {
		*combinations = append(*combinations, append([]int{}, combination...))
		return
	}

	for i := index; i < n; i++ {
		combination = append(combination, i)
		generateCombinationsHelper(n, k-1, i+1, combination, combinations)
		combination = combination[:len(combination)-1]
	}
}
