package api

import (
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
)

func GetPictures(c *gin.Context) {
	curLocation := c.Query("coordinates")
	url := "https://maps.googleapis.com/maps/api/streetview?size=400x400&location=" + curLocation + "&fov=80&heading=70&pitch=0&key=AIzaSyCo4C8j7kGJNFnr4hjK3KrANonXc5Dq56c"
	// fmt.Println(url)

	signedURL, err := SignURL(url, os.Getenv("SECRET"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "resource ID is required",
		})
		return
	}
	c.IndentedJSON(http.StatusOK, signedURL)
}
