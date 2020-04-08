#pragma once 
#include <GLFW/glfw3.h>

class EventState
{
public:
    EventState(GLFWwindow* w) : owningWindow(w) {}
    virtual ~EventState() = default;

    GLFWwindow* owningWindow;

private:
    EventState() = default;
};

class MouseState : public EventState
{
public:
    MouseState(GLFWwindow* w) : EventState(w), xPos(0.0), yPos(0.0), button(-1), action(-1), mods(-1), entered(false), left(false), scrollUp(false), scrollDown(false) {}
    virtual ~MouseState() = default;

    double xPos, yPos;
    int button, action, mods;
    bool entered, left, scrollUp, scrollDown;
};

class KeyboardState : public EventState
{
public:
    KeyboardState(GLFWwindow* w) : EventState(w) {}
    virtual ~KeyboardState() = default;

};

class WindowState : public EventState
{
public:
    WindowState(GLFWwindow* w) : EventState(w), resized(false), width(0), height(0) {}
    virtual ~WindowState() = default;

    bool  resized;
    int width, height;
};
class AggregateState : public EventState
{
public:
    AggregateState(GLFWwindow* w) : EventState(w), mouse(w), keyboard(w), window(w) {}
    virtual ~AggregateState() = default;

    void pullForward(const AggregateState& rhs)
    {
        mouse = rhs.mouse;
        keyboard = rhs.keyboard;
        window = rhs.window;
    }

    MouseState mouse;
    KeyboardState keyboard;
    WindowState window;
};

