#include <memory>

class VulkanInterfacePrivate;
class VulkanInterface
{
public:
    VulkanInterface();
    virtual ~VulkanInterface();

    void run();

protected:
    std::unique_ptr<VulkanInterfacePrivate> m_p;
};