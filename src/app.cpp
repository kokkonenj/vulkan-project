#include "app.h"
#include <SDL.h>
#include <SDL_vulkan.h>

void App::init()
{
	// Initialize SDL
	SDL_Init(SDL_INIT_VIDEO);
	SDL_WindowFlags windowFlags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

	// Create SDL window
	window = SDL_CreateWindow(
		"vulkan-project",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		windowExtent.width,
		windowExtent.height,
		windowFlags
	);

	isInitialized = true;
}

void App::cleanup()
{
	if (isInitialized)
	{
		SDL_DestroyWindow(window);
	}
}

void App::run()
{
	SDL_Event e;
	bool quitSignal = false;
	
	//main loop
	while (!quitSignal)
	{
		while (SDL_PollEvent(&e) != 0)
		{
			if (e.type == SDL_QUIT) quitSignal = true;
		}
	}
}
