package main

import (
	"net/http"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
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
	// fmt.Println(SignURL("https://maps.googleapis.com/maps/api/streetview?size=400x400&location=51.0899296,-113.9803&fov=80&heading=70&pitch=0&key=AIzaSyCo4C8j7kGJNFnr4hjK3KrANonXc5Dq56c", "gy3J-mcPumwIoQvr_KUjsFjNm-Y="))
}

func main() {
	router := gin.Default()

	router.Use(cors.New(cors.Config{
        AllowOrigins:     []string{"http://localhost:3000"}, // React app origin
        AllowMethods:     []string{"GET", "POST", "PUT", "DELETE"},
        AllowHeaders:     []string{"Origin", "Content-Type", "Accept"},
        AllowCredentials: true,
    }))
	router.GET("/locations", getLocations)

	router.Run(":8080")
}
