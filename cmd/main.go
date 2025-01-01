package main

import (
	"log"
	"net/http"
	"os"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
	"github.com/mliu1204/perceptioserver.git/cmd/api"
)


type location struct {
	ID      string `json:"id"`
	Address string `json:"address"`
	Rating  int    `json:"rating"`
}

var locations = []location{
	{ID: "1", Address: "76 Whiterm Gate NE", Rating: 10},
	{ID: "2", Address: "236 Woodside Bay SW", Rating: 10},
	{ID: "3", Address: "450 Jane Stanford Way", Rating: 10},
}

func getLocations(c *gin.Context) {
	c.IndentedJSON(http.StatusOK, locations)
}

func getPictures(c *gin.Context){
	curLocation := c.Query("coordinates")
	url := "https://maps.googleapis.com/maps/api/streetview?size=400x400&location=" + curLocation + "&fov=80&heading=70&pitch=0&key=AIzaSyCo4C8j7kGJNFnr4hjK3KrANonXc5Dq56c"
	// fmt.Println(url)
	
	signedURL, err := api.SignURL(url, os.Getenv("SECRET"))
	if (err != nil){
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "resource ID is required",
		})
		return
	}  
	c.IndentedJSON(http.StatusOK, signedURL)
}

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	router := gin.Default()

	//allow for communcation between front and backend through bypassing CORS protocol
	router.Use(cors.New(cors.Config{
        AllowOrigins:     []string{"http://localhost:3000"}, 
        AllowMethods:     []string{"GET", "POST", "PUT", "DELETE"},
        AllowHeaders:     []string{"Origin", "Content-Type", "Accept"},
        AllowCredentials: true,
    }))
	
	router.GET("/locations", getLocations)
	router.GET("/pictures", getPictures)

	router.Run(":8080")
}
