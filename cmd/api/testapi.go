package api

import (
	"net/http"

	"github.com/gin-gonic/gin"
)


type location struct {
	ID      string `json:"id"`
	Address string `json:"address"`
	Rating  int    `json:"rating"`
}

var Locations = []location{
	{ID: "1", Address: "76 Whiterm Gate NE", Rating: 10},
	{ID: "2", Address: "236 Woodside Bay SW", Rating: 10},
	{ID: "3", Address: "450 Jane Stanford Way", Rating: 10},
}


func GetLocations(c *gin.Context) {
	c.IndentedJSON(http.StatusOK, Locations)
}
